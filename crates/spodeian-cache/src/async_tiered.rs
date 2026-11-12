//! Asynchronous, write-through tiered store (`AsyncTieredStore`).
//!
//! **Stability: Stable** — semver-guaranteed since v1.1
//!
//! Like [`crate::TieredMemoryStore`], it keeps a bounded working set in memory and a
//! durable fallback tier behind it. It differs in three ways:
//!
//! - **Asynchronous**, so the browser fallback can be IndexedDB.
//! - **Write-through**: every `put` reaches the fallback tier before it returns, so the
//!   whole store survives a reload or restart. Memory is only a bounded read cache.
//! - **Interior mutability**: methods take `&self` and never hold a borrow across an
//!   `.await`, so one store can be shared (for example in an `Rc`) by concurrent tasks on a
//!   single thread.

use crate::{BoundedMemoryCache, CacheError, TieredMemoryMetrics};
use serde::{Serialize, de::DeserializeOwned};
use std::cell::RefCell;
use std::collections::BTreeSet;

#[cfg(not(target_arch = "wasm32"))]
const DISK_REF_PREFIX: &str = "spodeian-async-tiered/";

enum Fallback {
    None,
    #[cfg(not(target_arch = "wasm32"))]
    Disk(crate::ContentAddressedStorage),
    #[cfg(target_arch = "wasm32")]
    IndexedDb(crate::web::IndexedDbStore),
}

/// Bounded in-memory working set over a durable, write-through fallback tier.
///
/// # Examples
///
/// ```rust
/// # pollster::block_on(async {
/// use spodeian_cache::AsyncTieredStore;
///
/// let dir = std::env::temp_dir().join("spodeian_doc_async_tiered");
/// # let _ = std::fs::remove_dir_all(&dir);
/// let store = AsyncTieredStore::<String>::with_disk(2, &dir).unwrap();
/// for key in ["a", "b", "c"] {
///     store.put(key, key.to_uppercase()).await.unwrap();
/// }
/// // Only two items fit in memory, but all three are durable.
/// assert_eq!(store.metrics().memory_items_count, 2);
/// assert_eq!(store.keys(), vec!["a", "b", "c"]);
///
/// let reopened = AsyncTieredStore::<String>::with_disk(2, &dir).unwrap();
/// assert_eq!(reopened.get("a").await.unwrap(), Some("A".to_string()));
/// # let _ = std::fs::remove_dir_all(&dir);
/// # });
/// ```
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub struct AsyncTieredStore<V> {
    memory: RefCell<BoundedMemoryCache<String, V>>,
    fallback: Fallback,
    fallback_keys: RefCell<BTreeSet<String>>,
    metrics: RefCell<TieredMemoryMetrics>,
}

impl<V> std::fmt::Debug for AsyncTieredStore<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncTieredStore")
            .field("metrics", &*self.metrics.borrow())
            .finish_non_exhaustive()
    }
}

impl<V: Serialize + DeserializeOwned + Clone> AsyncTieredStore<V> {
    fn build(capacity: usize, fallback: Fallback, keys: BTreeSet<String>) -> Self {
        let store = Self {
            memory: RefCell::new(BoundedMemoryCache::new(capacity)),
            fallback,
            fallback_keys: RefCell::new(keys),
            metrics: RefCell::new(TieredMemoryMetrics::default()),
        };
        store.update_counts();
        store
    }

    /// A store with no fallback tier: items past `capacity` are dropped.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn in_memory(capacity: usize) -> Self {
        Self::build(capacity, Fallback::None, BTreeSet::new())
    }

    /// A store whose fallback tier is a content-addressed directory on disk. Items stored
    /// by an earlier store on the same directory are available immediately.
    #[cfg(not(target_arch = "wasm32"))]
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn with_disk(
        capacity: usize,
        dir: impl AsRef<std::path::Path>,
    ) -> Result<Self, CacheError> {
        let cas = crate::ContentAddressedStorage::new(dir)?;
        let keys = cas
            .list_names()?
            .into_iter()
            .filter_map(|name| name.strip_prefix(DISK_REF_PREFIX).map(str::to_string))
            .collect();
        Ok(Self::build(capacity, Fallback::Disk(cas), keys))
    }

    /// A store whose fallback tier is an IndexedDB object store, which it owns entirely.
    /// Items stored by an earlier session are available immediately.
    #[cfg(target_arch = "wasm32")]
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn with_indexed_db(
        capacity: usize,
        store: crate::web::IndexedDbStore,
    ) -> Result<Self, CacheError> {
        let keys = store.keys().await?.into_iter().collect();
        Ok(Self::build(capacity, Fallback::IndexedDb(store), keys))
    }

    async fn write_fallback(&self, key: &str, value: &V) -> Result<(), CacheError> {
        match &self.fallback {
            Fallback::None => return Ok(()),
            #[cfg(not(target_arch = "wasm32"))]
            Fallback::Disk(cas) => {
                cas.put_named_json(&format!("{DISK_REF_PREFIX}{key}"), value)?;
            }
            #[cfg(target_arch = "wasm32")]
            Fallback::IndexedDb(store) => store.put_json(key, value).await?,
        }
        self.fallback_keys.borrow_mut().insert(key.to_string());
        Ok(())
    }

    async fn read_fallback(&self, key: &str) -> Result<Option<V>, CacheError> {
        if !self.fallback_keys.borrow().contains(key) {
            return Ok(None);
        }
        match &self.fallback {
            Fallback::None => Ok(None),
            #[cfg(not(target_arch = "wasm32"))]
            Fallback::Disk(cas) => match cas.get_named_json(&format!("{DISK_REF_PREFIX}{key}")) {
                Ok(value) => Ok(Some(value)),
                Err(CacheError::NotFound(_)) => Ok(None),
                Err(e) => Err(e),
            },
            #[cfg(target_arch = "wasm32")]
            Fallback::IndexedDb(store) => store.get_json(key).await,
        }
    }

    async fn delete_fallback(&self, key: &str) -> Result<bool, CacheError> {
        if !self.fallback_keys.borrow().contains(key) {
            return Ok(false);
        }
        match &self.fallback {
            Fallback::None => {}
            #[cfg(not(target_arch = "wasm32"))]
            Fallback::Disk(cas) => {
                cas.remove_named(&format!("{DISK_REF_PREFIX}{key}"))?;
            }
            #[cfg(target_arch = "wasm32")]
            Fallback::IndexedDb(store) => store.delete(key).await?,
        }
        self.fallback_keys.borrow_mut().remove(key);
        Ok(true)
    }

    fn cache_in_memory(&self, key: String, value: V) {
        let evicted = self.memory.borrow_mut().put(key, value);
        if evicted.is_some() {
            self.metrics.borrow_mut().evictions_to_fallback += 1;
        }
    }

    /// Stores `value` under `key`. Returns once the value is durable in the fallback tier;
    /// on error nothing is changed.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn put(&self, key: impl Into<String>, value: V) -> Result<(), CacheError> {
        let key = key.into();
        self.write_fallback(&key, &value).await?;
        self.cache_in_memory(key, value);
        self.update_counts();
        Ok(())
    }

    /// Retrieves the value under `key`, from memory if it is in the working set, otherwise
    /// from the fallback tier (which then promotes it into memory).
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn get(&self, key: &str) -> Result<Option<V>, CacheError> {
        let in_memory = self.memory.borrow_mut().get(&key.to_string()).cloned();
        if let Some(value) = in_memory {
            self.metrics.borrow_mut().memory_hits += 1;
            return Ok(Some(value));
        }
        let Some(value) = self.read_fallback(key).await? else {
            return Ok(None);
        };
        self.metrics.borrow_mut().fallback_hits += 1;
        self.cache_in_memory(key.to_string(), value.clone());
        self.update_counts();
        Ok(Some(value))
    }

    /// Removes `key` from both tiers. Returns whether it existed.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn remove(&self, key: &str) -> Result<bool, CacheError> {
        let from_fallback = self.delete_fallback(key).await?;
        let from_memory = self.memory.borrow_mut().remove(&key.to_string()).is_some();
        self.update_counts();
        Ok(from_fallback || from_memory)
    }

    /// Whether `key` exists in either tier.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn contains(&self, key: &str) -> bool {
        self.memory.borrow().contains_key(&key.to_string())
            || self.fallback_keys.borrow().contains(key)
    }

    /// Every key in the store, sorted.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn keys(&self) -> Vec<String> {
        let mut keys: BTreeSet<String> = self.fallback_keys.borrow().clone();
        keys.extend(self.memory.borrow().iter().map(|(k, _)| k.clone()));
        keys.into_iter().collect()
    }

    /// Number of distinct keys in the store.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn len(&self) -> usize {
        self.keys().len()
    }

    /// Whether the store holds no items.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hit, eviction and occupancy counters.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn metrics(&self) -> TieredMemoryMetrics {
        *self.metrics.borrow()
    }

    /// The configured in-memory capacity.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn memory_capacity(&self) -> usize {
        self.memory.borrow().capacity()
    }

    fn update_counts(&self) {
        let memory = self.memory.borrow().len();
        let fallback = self.fallback_keys.borrow().len();
        let mut metrics = self.metrics.borrow_mut();
        metrics.memory_items_count = memory;
        metrics.fallback_items_count = fallback;
    }
}
