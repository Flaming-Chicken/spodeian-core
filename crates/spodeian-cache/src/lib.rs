//! Preferential Multi-Tier Caching & Persistence Engine (`spodeian-cache`).
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides intelligent storage routing across:
//! - **Cache API (`caches`)**: Dedicated to large, immutable binary assets (`.safetensors`, `.wasm`, `.parquet`, WGSL compute pipelines, audio buffers).
//! - **IndexedDB (`idb`)**: Dedicated to structured, relational, transaction-safe records (user states, incremental OHLCV ticks, factor loadings, replay memories).
//! - **LocalStorage**: Dedicated to fast, synchronous configuration toggles (< 16 KB).
//! - **Native CAS**: Cryptographic Content-Addressed Storage (SHA-256) for native desktop and mobile.
pub mod async_tiered;
pub mod preferential_store;
pub mod tiered_memory;
#[cfg(target_arch = "wasm32")]
pub mod web;

#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub use async_tiered::AsyncTieredStore;
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub use preferential_store::PreferentialStore;

#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use tiered_memory::{BoundedMemoryCache, TieredMemoryMetrics, TieredMemoryStore};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization failure: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error(
        "Cryptographic tamper detected for key '{key}': expected {expected}, computed {computed}"
    )]
    TamperDetected {
        key: String,
        expected: String,
        computed: String,
    },
    #[error("Item not found: {0}")]
    NotFound(String),
    #[error("Browser storage failure: {0}")]
    BrowserStorage(String),
}

/// Storage tier designation for preferential caching.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// Browser Cache API (or Native CAS) for large immutable binary blobs (> 512 KB, SafeTensors, WASM)
    CacheApi,
    /// IndexedDB (or Native relational store) for structured queryable entities and replay buffers
    IndexedDb,
    /// Fast synchronous LocalStorage for small session keys (< 16 KB)
    LocalStorage,
    /// Native disk Content-Addressed Storage
    NativeCas,
}

impl StorageTier {
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn label(self) -> &'static str {
        match self {
            Self::CacheApi => "Cache API (Immutable Binary Tier)",
            Self::IndexedDb => "IndexedDB (Structured Relational Tier)",
            Self::LocalStorage => "LocalStorage (Fast Session Tier)",
            Self::NativeCas => "Native CAS (Content-Addressed Disk Tier)",
        }
    }
}

/// Metadata header describing cached assets.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub key: String,
    pub sha256_hash: String,
    pub byte_size: usize,
    pub tier: StorageTier,
    pub content_type: String,
    pub is_immutable: bool,
}

/// Preferential router deciding the optimal storage backend based on asset properties.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub struct PreferentialRouter;

impl PreferentialRouter {
    /// Determines the optimal storage tier for an asset.
    ///
    /// Rules:
    /// 1. Payloads $> 512\text{ KB}$ or binary mime types $\to$ `CacheApi` (on web) or `NativeCas` (on native).
    /// 2. Immutable assets (`.safetensors`, `.wasm`, `.parquet`) $\to$ `CacheApi` / `NativeCas`.
    /// 3. Structured relational models $\to$ `IndexedDb`.
    /// 4. Tiny config flags $< 16\text{ KB}$ $\to$ `LocalStorage`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn determine_tier(
        size_bytes: usize,
        content_type: &str,
        is_immutable: bool,
    ) -> StorageTier {
        #[cfg(target_arch = "wasm32")]
        {
            if is_immutable
                || size_bytes > 512 * 1024
                || content_type.starts_with("model/")
                || content_type.contains("octet-stream")
            {
                StorageTier::CacheApi
            } else if size_bytes < 16 * 1024 && !content_type.contains("json") {
                StorageTier::LocalStorage
            } else {
                StorageTier::IndexedDb
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (content_type, is_immutable, size_bytes);
            StorageTier::NativeCas
        }
    }
}

/// Cryptographic Content-Addressed Storage (CAS) for native desktop, mobile, and offline runtimes.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone)]
pub struct ContentAddressedStorage {
    root_dir: PathBuf,
}

impl ContentAddressedStorage {
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn new(root_dir: impl AsRef<Path>) -> Result<Self, CacheError> {
        let path = root_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;
        Ok(Self { root_dir: path })
    }

    /// Computes the cryptographic SHA-256 hash of bytes.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn compute_sha256(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    /// Stores raw binary data addressed by its SHA-256 hash.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn put(&self, data: &[u8]) -> Result<String, CacheError> {
        let hash = Self::compute_sha256(data);
        let file_path = self.root_dir.join(&hash);
        if !file_path.exists() {
            std::fs::write(&file_path, data)?;
        }
        Ok(hash)
    }

    /// Retrieves data by its SHA-256 hash, automatically verifying cryptographic integrity.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn get(&self, hash: &str) -> Result<Vec<u8>, CacheError> {
        let file_path = self.root_dir.join(hash);
        if !file_path.exists() {
            return Err(CacheError::NotFound(hash.to_string()));
        }

        let bytes = std::fs::read(&file_path)?;
        let computed = Self::compute_sha256(&bytes);
        if computed != hash {
            return Err(CacheError::TamperDetected {
                key: hash.to_string(),
                expected: hash.to_string(),
                computed,
            });
        }

        Ok(bytes)
    }

    /// Stores a serializable entity as JSON, returning its content hash.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn put_json<T: Serialize>(&self, item: &T) -> Result<String, CacheError> {
        let json_bytes = serde_json::to_vec(item)?;
        self.put(&json_bytes)
    }

    /// Retrieves a serializable entity by hash, with integrity verification and deserialization.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn get_json<T: DeserializeOwned>(&self, hash: &str) -> Result<T, CacheError> {
        let bytes = self.get(hash)?;
        let item = serde_json::from_slice(&bytes)?;
        Ok(item)
    }

    /// Checks if a given content hash exists in the cache.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn contains(&self, hash: &str) -> bool {
        self.root_dir.join(hash).exists()
    }

    // ------------------------------------------------------------------
    // Named references: stable keys that point at content hashes, so data
    // stored in one session can be found again after a restart.
    // ------------------------------------------------------------------

    fn refs_dir(&self) -> PathBuf {
        self.root_dir.join("refs")
    }

    fn ref_path(&self, name: &str) -> PathBuf {
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(name.as_bytes());
        self.refs_dir().join(encoded)
    }

    /// Stores data and points `name` at it, replacing any previous target.
    ///
    /// Returns the content hash. Names may contain any characters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use spodeian_cache::ContentAddressedStorage;
    ///
    /// let dir = std::env::temp_dir().join("spodeian_doc_named_refs");
    /// let cas = ContentAddressedStorage::new(&dir).unwrap();
    /// cas.put_named("settings/theme", b"dark").unwrap();
    ///
    /// // A fresh handle on the same directory finds the data by name.
    /// let reopened = ContentAddressedStorage::new(&dir).unwrap();
    /// assert_eq!(reopened.get_named("settings/theme").unwrap(), b"dark");
    /// # let _ = std::fs::remove_dir_all(&dir);
    /// ```
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn put_named(&self, name: &str, data: &[u8]) -> Result<String, CacheError> {
        let hash = self.put(data)?;
        std::fs::create_dir_all(self.refs_dir())?;
        let target = self.ref_path(name);
        // Write then rename, so a crash never leaves a half-written reference.
        let staging = target.with_extension("tmp");
        std::fs::write(&staging, hash.as_bytes())?;
        std::fs::rename(&staging, &target)?;
        Ok(hash)
    }

    /// Returns the content hash `name` points at, if the reference exists.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn resolve_name(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.ref_path(name))
            .ok()
            .map(|s| s.trim().to_string())
    }

    /// Retrieves the data `name` points at, verifying its integrity.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn get_named(&self, name: &str) -> Result<Vec<u8>, CacheError> {
        let hash = self
            .resolve_name(name)
            .ok_or_else(|| CacheError::NotFound(name.to_string()))?;
        self.get(&hash)
    }

    /// Stores a serializable entity as JSON under `name`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn put_named_json<T: Serialize>(&self, name: &str, item: &T) -> Result<String, CacheError> {
        let json_bytes = serde_json::to_vec(item)?;
        self.put_named(name, &json_bytes)
    }

    /// Retrieves and deserializes the JSON entity stored under `name`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn get_named_json<T: DeserializeOwned>(&self, name: &str) -> Result<T, CacheError> {
        let bytes = self.get_named(name)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Removes the reference `name`; the content itself is left in place, since other
    /// references may share it. Returns whether a reference existed.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn remove_named(&self, name: &str) -> Result<bool, CacheError> {
        match std::fs::remove_file(self.ref_path(name)) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Lists every reference name, in no particular order.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn list_names(&self) -> Result<Vec<String>, CacheError> {
        use base64::Engine as _;
        let dir = match std::fs::read_dir(self.refs_dir()) {
            Ok(dir) => dir,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let mut names = Vec::new();
        for entry in dir {
            let file_name = entry?.file_name();
            let Some(encoded) = file_name.to_str() else {
                continue;
            };
            if encoded.ends_with(".tmp") {
                continue;
            }
            if let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(encoded)
                && let Ok(name) = String::from_utf8(bytes)
            {
                names.push(name);
            }
        }
        Ok(names)
    }
}
