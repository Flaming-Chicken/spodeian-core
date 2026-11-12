//! Storage facade that carries out [`crate::PreferentialRouter`]'s decisions.
//!
//! **Stability: Stable** — semver-guaranteed since v1.1
//!
//! On the web, each value goes to the tier the router picks (local storage, IndexedDB or
//! the Cache API) and is removed from the other tiers, so a later load never returns a
//! stale copy. On desktop and mobile, values go to named references in content-addressed
//! storage.

use crate::{CacheError, PreferentialRouter, StorageTier};
use serde::{Serialize, de::DeserializeOwned};

/// Reference-name prefix for native storage.
#[cfg(not(target_arch = "wasm32"))]
const NATIVE_PREFIX: &str = "spodeian-preferential/";

/// Routes each save to the storage tier [`PreferentialRouter`] chooses for it.
///
/// # Examples
///
/// ```rust
/// # pollster::block_on(async {
/// use spodeian_cache::{PreferentialStore, StorageTier};
///
/// let dir = std::env::temp_dir().join("spodeian_doc_preferential");
/// let store = PreferentialStore::open_native(&dir).unwrap();
/// let tier = store.save_json("settings", &vec!["dark"]).await.unwrap();
/// assert_eq!(tier, StorageTier::NativeCas);
/// let loaded: Option<Vec<String>> = store.load_json("settings").await.unwrap();
/// assert_eq!(loaded, Some(vec!["dark".to_string()]));
/// # let _ = std::fs::remove_dir_all(&dir);
/// # });
/// ```
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
#[derive(Debug, Clone)]
pub struct PreferentialStore {
    #[cfg(target_arch = "wasm32")]
    local: crate::web::LocalStorageStore,
    #[cfg(target_arch = "wasm32")]
    indexed_db: crate::web::IndexedDbStore,
    #[cfg(target_arch = "wasm32")]
    cache: crate::web::CacheApiStore,
    #[cfg(not(target_arch = "wasm32"))]
    cas: crate::ContentAddressedStorage,
}

impl PreferentialStore {
    /// Opens the browser tiers for `namespace`: the IndexedDB database and Cache API cache
    /// `spodeian-{namespace}`, and local storage keys prefixed `spodeian-{namespace}:`.
    #[cfg(target_arch = "wasm32")]
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn open_web(namespace: &str) -> Result<Self, CacheError> {
        let name = format!("spodeian-{namespace}");
        Ok(Self {
            local: crate::web::LocalStorageStore::open(&format!("{name}:"))?,
            indexed_db: crate::web::IndexedDbStore::open(&name, "records").await?,
            cache: crate::web::CacheApiStore::open(&name).await?,
        })
    }

    /// Opens native storage in `dir`.
    #[cfg(not(target_arch = "wasm32"))]
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn open_native(dir: impl AsRef<std::path::Path>) -> Result<Self, CacheError> {
        Ok(Self {
            cas: crate::ContentAddressedStorage::new(dir)?,
        })
    }

    /// Saves `bytes` under `key` in the tier the router chooses, returning that tier.
    ///
    /// On the web, copies of `key` in other tiers are removed afterwards; an error there is
    /// returned so the caller can retry rather than leave a stale copy behind.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn save_bytes(
        &self,
        key: &str,
        bytes: &[u8],
        content_type: &str,
        is_immutable: bool,
    ) -> Result<StorageTier, CacheError> {
        let tier = PreferentialRouter::determine_tier(bytes.len(), content_type, is_immutable);

        #[cfg(target_arch = "wasm32")]
        {
            use base64::Engine as _;
            match tier {
                StorageTier::LocalStorage => {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                    self.local.set(key, &encoded)?;
                }
                StorageTier::IndexedDb => self.indexed_db.put_bytes(key, bytes).await?,
                StorageTier::CacheApi => self.cache.put_bytes(key, bytes, content_type).await?,
                StorageTier::NativeCas => unreachable!("the router never picks NativeCas on wasm"),
            }
            if tier != StorageTier::LocalStorage {
                self.local.remove(key)?;
            }
            if tier != StorageTier::IndexedDb {
                self.indexed_db.delete(key).await?;
            }
            if tier != StorageTier::CacheApi {
                self.cache.delete(key).await?;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = content_type;
            self.cas
                .put_named(&format!("{NATIVE_PREFIX}{key}"), bytes)?;
        }

        Ok(tier)
    }

    /// Loads the bytes stored under `key`, from whichever tier holds them.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn load_bytes(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        #[cfg(target_arch = "wasm32")]
        {
            use base64::Engine as _;
            if let Some(encoded) = self.local.get(key)? {
                return base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map(Some)
                    .map_err(|e| CacheError::BrowserStorage(format!("corrupt value: {e}")));
            }
            if let Some(bytes) = self.indexed_db.get_bytes(key).await? {
                return Ok(Some(bytes));
            }
            self.cache.get_bytes(key).await
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            match self.cas.get_named(&format!("{NATIVE_PREFIX}{key}")) {
                Ok(bytes) => Ok(Some(bytes)),
                Err(CacheError::NotFound(_)) => Ok(None),
                Err(e) => Err(e),
            }
        }
    }

    /// Removes `key` from every tier. Returns whether it existed.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn remove(&self, key: &str) -> Result<bool, CacheError> {
        #[cfg(target_arch = "wasm32")]
        {
            let existed = self.load_bytes(key).await?.is_some();
            self.local.remove(key)?;
            self.indexed_db.delete(key).await?;
            self.cache.delete(key).await?;
            Ok(existed)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.cas.remove_named(&format!("{NATIVE_PREFIX}{key}"))
        }
    }

    /// Saves a serializable value as JSON, returning the tier used.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn save_json<T: Serialize>(
        &self,
        key: &str,
        value: &T,
    ) -> Result<StorageTier, CacheError> {
        let bytes = serde_json::to_vec(value)?;
        self.save_bytes(key, &bytes, "application/json", false)
            .await
    }

    /// Loads and deserializes a JSON value.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn load_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, CacheError> {
        match self.load_bytes(key).await? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }
}
