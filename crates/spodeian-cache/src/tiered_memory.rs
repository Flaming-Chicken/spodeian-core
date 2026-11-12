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
///
/// # Examples
///
/// ```rust
/// use spodeian_cache::BoundedMemoryCache;
///
/// let mut cache = BoundedMemoryCache::new(2);
/// cache.put("a", 1);
/// cache.put("b", 2);
/// assert_eq!(cache.get(&"a"), Some(&1));
///
/// // Pushing "c" exceeds capacity, evicting the least recently used ("b")
/// let evicted = cache.put("c", 3);
/// assert_eq!(evicted, Some(("b", 2)));
/// assert_eq!(cache.get(&"b"), None);
/// ```
#[derive(Debug, Clone)]
pub struct BoundedMemoryCache<K, V> {
    capacity: usize,
    entries: HashMap<K, (V, u64)>,
    order: VecDeque<(K, u64)>,
    access_counter: u64,
}

impl<K: Hash + Eq + Clone, V> BoundedMemoryCache<K, V> {
    /// Creates a new in-memory cache with the specified maximum item capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::with_capacity(capacity.min(1024)),
            order: VecDeque::with_capacity(capacity.min(1024)),
            access_counter: 0,
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

    /// Retrieves an item from memory, promoting it to the most-recently-used position in O(1) time.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.access_counter = self.access_counter.wrapping_add(1);
        let current_access = self.access_counter;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.1 = current_access;
            self.order.push_back((key.clone(), current_access));
            Some(&entry.0)
        } else {
            None
        }
    }

    /// Retrieves an item from memory immutably without altering LRU order.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|(v, _)| v)
    }

    /// Retrieves a mutable reference to an item, promoting it to the MRU position in O(1) time.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.access_counter = self.access_counter.wrapping_add(1);
        let current_access = self.access_counter;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.1 = current_access;
            self.order.push_back((key.clone(), current_access));
            Some(&mut entry.0)
        } else {
            None
        }
    }

    /// Inserts a key-value pair into the in-memory cache in O(1) time.
    ///
    /// If inserting causes the cache to exceed its capacity, the least-recently-used (LRU)
    /// item is evicted and returned as `Some((evicted_key, evicted_value))`.
    pub fn put(&mut self, key: K, value: V) -> Option<(K, V)> {
        self.access_counter = self.access_counter.wrapping_add(1);
        let current_access = self.access_counter;
        let mut evicted = None;

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.0 = value;
            entry.1 = current_access;
            self.order.push_back((key, current_access));
        } else {
            while self.entries.len() >= self.capacity {
                if let Some((candidate_key, candidate_access)) = self.order.pop_front() {
                    if let Some((_, entry_access)) = self.entries.get(&candidate_key)
                        && *entry_access == candidate_access
                        && let Some((lru_val, _)) = self.entries.remove(&candidate_key)
                    {
                        evicted = Some((candidate_key, lru_val));
                        break;
                    }
                } else {
                    break;
                }
            }

            self.order.push_back((key.clone(), current_access));
            self.entries.insert(key, (value, current_access));
        }

        if self.order.len() > self.capacity * 4 {
            self.compact_order();
        }

        evicted
    }

    /// Removes an item from the in-memory cache in O(1) time.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key).map(|(val, _)| val)
    }

    /// Clears all items from the in-memory cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.access_counter = 0;
    }

    /// Iterates over all current in-memory entries.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, (v, _))| (k, v))
    }

    fn compact_order(&mut self) {
        let entries = &self.entries;
        self.order
            .retain(|(k, acc)| entries.get(k).map(|(_, a)| *a == *acc).unwrap_or(false));
    }
}

/// Name prefix for the references a [`TieredMemoryStore`] keeps in its fallback directory.
const FALLBACK_REF_PREFIX: &str = "spodeian-tiered-memory/";

/// Tiered memory store coordinating bounded in-memory caching with a persistent fallback tier.
///
/// Ensures high-frequency operations occur at in-memory speed, while offloading cold items
/// to a content-addressed directory on disk so memory does not overfill. For the browser,
/// where storage is asynchronous, see `AsyncTieredStore`.
///
/// Since v1.1 the fallback index is itself persisted as named references, so items that
/// spilled to disk are found again when a store is reopened on the same directory. Items
/// still in memory are not persisted until they spill or [`TieredMemoryStore::flush`] runs.
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

        // Rebuild the fallback index from references left by a previous session.
        let mut fallback_index = HashMap::new();
        if let Some(cas) = &fallback_cas {
            for name in cas.list_names()? {
                if let Some(key) = name.strip_prefix(FALLBACK_REF_PREFIX)
                    && let Some(hash) = cas.resolve_name(&name)
                {
                    fallback_index.insert(key.to_string(), hash);
                }
            }
        }

        let mut store = Self {
            memory: BoundedMemoryCache::new(capacity),
            fallback_cas,
            fallback_index,
            metrics: TieredMemoryMetrics {
                memory_items_count: 0,
                fallback_items_count: 0,
                ..Default::default()
            },
        };
        store.update_counts();
        Ok(store)
    }

    /// Inserts an item into memory. If the in-memory capacity is reached, the oldest item is
    /// evicted and written to the persistent fallback tier to prevent memory overfilling.
    pub fn put(&mut self, key: impl Into<String>, value: V) -> Result<(), CacheError> {
        let k = key.into();

        // If the key was previously in fallback, remove its stale fallback entry
        if self.fallback_index.remove(&k).is_some() {
            self.forget_fallback_ref(&k)?;
        }

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
            self.forget_fallback_ref(key)?;
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
    ///
    /// Removing the persisted reference is best-effort; a failure there leaves a stale
    /// reference that is replaced the next time the key is stored.
    pub fn remove(&mut self, key: &str) -> bool {
        let removed_mem = self.memory.remove(&key.to_string()).is_some();
        let removed_fb = self.fallback_index.remove(key).is_some();
        if removed_fb {
            let _ = self.forget_fallback_ref(key);
        }
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

    /// Writes every in-memory item to the fallback tier without evicting it, so the whole
    /// store survives a restart. Returns the number of items written; 0 without a fallback.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn flush(&mut self) -> Result<usize, CacheError> {
        let Some(cas) = &self.fallback_cas else {
            return Ok(0);
        };
        let mut written = 0;
        for (key, value) in self.memory.iter() {
            let hash = cas.put_named_json(&format!("{FALLBACK_REF_PREFIX}{key}"), value)?;
            self.fallback_index.insert(key.clone(), hash);
            written += 1;
        }
        self.update_counts();
        Ok(written)
    }

    fn persist_to_fallback(&mut self, key: String, value: V) -> Result<(), CacheError> {
        if let Some(cas) = &self.fallback_cas {
            let hash = cas.put_named_json(&format!("{FALLBACK_REF_PREFIX}{key}"), &value)?;
            self.fallback_index.insert(key, hash);
        }
        Ok(())
    }

    fn forget_fallback_ref(&self, key: &str) -> Result<(), CacheError> {
        if let Some(cas) = &self.fallback_cas {
            cas.remove_named(&format!("{FALLBACK_REF_PREFIX}{key}"))?;
        }
        Ok(())
    }

    fn update_counts(&mut self) {
        self.metrics.memory_items_count = self.memory.len();
        self.metrics.fallback_items_count = self.fallback_index.len();
    }
}
