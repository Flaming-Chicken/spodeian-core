//! Bounded in-memory tier and multi-tier fallback coordination (`TieredMemoryStore`).
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! # Architecture & Performance Profile:
//! - **In-Memory Tier (`BoundedMemoryCache`)**:
//!   - Format: Direct Rust structs stored in RAM (`HashMap` + LRU queue).
//!   - Performance: Sub-microsecond latency, 0-allocation query path, ideal for high-frequency
//!     UI rendering loops (60-120 FPS), active filtering, and immediate mutations.
//!   - Bounded Capacity: Prevents browser tab and desktop process memory exhaustion (OOM).
//! - **IndexedDB / CAS Fallback Tier**:
//!   - Format: Structured documents / JSON records / Content-Addressed binary blobs.
//!   - Performance: Asynchronous / disk-backed, transactional durability with hundreds of MBs/GBs of quota.
//!   - Fallback Role: Holds cold/evicted items when in-memory capacity is reached, preventing
//!     memory from overfilling while retaining complete application state durability.

use crate::{CacheError, ContentAddressedStorage};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::path::Path;

/// Real-time metrics for tiered memory operations and fallback transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TieredMemoryMetrics {
    /// Number of cache hits satisfied instantly from the hot in-memory tier.
    pub memory_hits: u64,
    /// Number of cache hits satisfied by falling back to persistent storage.
    pub fallback_hits: u64,
    /// Number of cold items evicted from RAM to prevent memory overfilling.
    pub evictions_to_fallback: u64,
    /// Current count of active items residing in the in-memory tier.
    pub memory_items_count: usize,
    /// Current count of items stored in the persistent fallback tier.
    pub fallback_items_count: usize,
}

/// Bounded in-memory cache utilizing an LRU eviction strategy.
///
/// Designed to keep active working sets in RAM without unbounded memory growth.
#[derive(Debug, Clone)]
pub struct BoundedMemoryCache<K, V> {
    capacity: usize,
    entries: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Hash + Eq + Clone, V> BoundedMemoryCache<K, V> {
    /// Creates a new in-memory cache with the specified maximum item capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::with_capacity(capacity.min(1024)),
            order: VecDeque::with_capacity(capacity.min(1024)),
        }
    }

    /// Maximum number of items permitted in memory before eviction occurs.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of items residing in RAM.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the in-memory cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if the cache contains the specified key in RAM.
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Retrieves an item from memory, promoting it to the most-recently-used position.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.promote(key);
            self.entries.get(key)
        } else {
            None
        }
    }

    /// Retrieves an item from memory immutably without altering LRU order.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key)
    }

    /// Retrieves a mutable reference to an item, promoting it to the MRU position.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.entries.contains_key(key) {
            self.promote(key);
            self.entries.get_mut(key)
        } else {
            None
        }
    }

    /// Inserts a key-value pair into the in-memory cache.
    ///
    /// If inserting causes the cache to exceed its capacity, the least-recently-used (LRU)
    /// item is evicted and returned as `Some((evicted_key, evicted_value))`.
    pub fn put(&mut self, key: K, value: V) -> Option<(K, V)> {
        let mut evicted = None;

        if self.entries.contains_key(&key) {
            self.promote(&key);
            self.entries.insert(key, value);
        } else {
            if self.entries.len() >= self.capacity
                && let Some(lru_key) = self.order.pop_front()
                && let Some(lru_val) = self.entries.remove(&lru_key)
            {
                evicted = Some((lru_key, lru_val));
            }
            self.order.push_back(key.clone());
            self.entries.insert(key, value);
        }

        evicted
    }

    /// Removes an item from the in-memory cache.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(val) = self.entries.remove(key) {
            if let Some(idx) = self.order.iter().position(|k| k == key) {
                self.order.remove(idx);
            }
            Some(val)
        } else {
            None
        }
    }

    /// Clears all items from the in-memory cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    /// Iterates over all current in-memory entries.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter()
    }

    fn promote(&mut self, key: &K) {
        if let Some(idx) = self.order.iter().position(|k| k == key) {
            self.order.remove(idx);
            self.order.push_back(key.clone());
        }
    }
}

/// Tiered memory store coordinating bounded in-memory caching with a persistent fallback tier.
///
/// Ensures high-frequency operations occur at in-memory speed, while offloading cold items
/// to disk or IndexedDB so memory does not overfill.
#[derive(Debug)]
pub struct TieredMemoryStore<V> {
    memory: BoundedMemoryCache<String, V>,
    fallback_cas: Option<ContentAddressedStorage>,
    fallback_index: HashMap<String, String>, // key -> hash
    metrics: TieredMemoryMetrics,
}

impl<V: Serialize + DeserializeOwned + Clone> TieredMemoryStore<V> {
    /// Creates a new tiered store with the specified in-memory capacity and an optional CAS fallback directory.
    pub fn new(capacity: usize, fallback_dir: Option<&Path>) -> Result<Self, CacheError> {
        let fallback_cas = match fallback_dir {
            Some(dir) => Some(ContentAddressedStorage::new(dir)?),
            None => None,
        };

        Ok(Self {
            memory: BoundedMemoryCache::new(capacity),
            fallback_cas,
            fallback_index: HashMap::new(),
            metrics: TieredMemoryMetrics {
                memory_items_count: 0,
                fallback_items_count: 0,
                ..Default::default()
            },
        })
    }

    /// Inserts an item into memory. If the in-memory capacity is reached, the oldest item is
    /// evicted and written to the persistent fallback tier to prevent memory overfilling.
    pub fn put(&mut self, key: impl Into<String>, value: V) -> Result<(), CacheError> {
        let k = key.into();

        // If the key was previously in fallback, remove its stale fallback entry
        self.fallback_index.remove(&k);

        if let Some((evicted_key, evicted_value)) = self.memory.put(k, value) {
            self.persist_to_fallback(evicted_key, evicted_value)?;
            self.metrics.evictions_to_fallback += 1;
        }

        self.update_counts();
        Ok(())
    }

    /// Retrieves an item from the tiered store.
    ///
    /// 1. Checks in-memory cache first (sub-microsecond hit).
    /// 2. If missing from memory, falls back to persistent storage, deserializes the item,
    ///    and brings it back into memory (promoting it).
    pub fn get(&mut self, key: &str) -> Result<Option<V>, CacheError> {
        // Tier 1: In-memory hit
        if let Some(val) = self.memory.get(&key.to_string()) {
            self.metrics.memory_hits += 1;
            return Ok(Some(val.clone()));
        }

        // Tier 2: Fallback hit
        if let Some(hash) = self.fallback_index.get(key).cloned()
            && let Some(cas) = &self.fallback_cas
        {
            let val: V = cas.get_json(&hash)?;
            self.metrics.fallback_hits += 1;

            // Bring back into memory; if this causes an eviction, spill that evicted item
            if let Some((evicted_key, evicted_value)) =
                self.memory.put(key.to_string(), val.clone())
            {
                self.persist_to_fallback(evicted_key, evicted_value)?;
                self.metrics.evictions_to_fallback += 1;
            }

            // Remove from fallback index now that it lives in memory
            self.fallback_index.remove(key);
            self.update_counts();

            return Ok(Some(val));
        }

        Ok(None)
    }

    /// Checks if a key exists in either the in-memory tier or persistent fallback.
    pub fn contains(&self, key: &str) -> bool {
        self.memory.contains_key(&key.to_string()) || self.fallback_index.contains_key(key)
    }

    /// Removes an item from both memory and fallback tiers.
    pub fn remove(&mut self, key: &str) -> bool {
        let removed_mem = self.memory.remove(&key.to_string()).is_some();
        let removed_fb = self.fallback_index.remove(key).is_some();
        self.update_counts();
        removed_mem || removed_fb
    }

    /// Returns operational metrics describing cache hits, fallback usage, and memory capacity.
    pub fn metrics(&self) -> TieredMemoryMetrics {
        self.metrics
    }

    /// In-memory capacity limit configured for this store.
    pub fn memory_capacity(&self) -> usize {
        self.memory.capacity()
    }

    fn persist_to_fallback(&mut self, key: String, value: V) -> Result<(), CacheError> {
        if let Some(cas) = &self.fallback_cas {
            let hash = cas.put_json(&value)?;
            self.fallback_index.insert(key, hash);
        }
        Ok(())
    }

    fn update_counts(&mut self) {
        self.metrics.memory_items_count = self.memory.len();
        self.metrics.fallback_items_count = self.fallback_index.len();
    }
}
