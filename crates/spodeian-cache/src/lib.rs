//! Preferential Multi-Tier Caching & Persistence Engine (`spodeian-cache`).
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides intelligent storage routing across:
//! - **Cache API (`caches`)**: Dedicated to large, immutable binary assets (`.safetensors`, `.wasm`, `.parquet`, WGSL compute pipelines, audio buffers).
//! - **IndexedDB (`idb`)**: Dedicated to structured, relational, transaction-safe records (user states, incremental OHLCV ticks, factor loadings, replay memories).
//! - **LocalStorage**: Dedicated to fast, synchronous configuration toggles (< 16 KB).
//! - **Native CAS**: Cryptographic Content-Addressed Storage (SHA-256) for native desktop and mobile.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
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
    #[error("Cryptographic tamper detected for key '{key}': expected {expected}, computed {computed}")]
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
    pub fn determine_tier(size_bytes: usize, content_type: &str, is_immutable: bool) -> StorageTier {
        #[cfg(target_arch = "wasm32")]
        {
            if is_immutable || size_bytes > 512 * 1024 || content_type.starts_with("model/") || content_type.contains("octet-stream") {
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
}
