//! Unified multi-tiered state persistence, atomic snapshot files with backup recovery,
//! versioned schema migration, and browser PWA storage diagnostics.
//!
//! Combines native atomic `.tmp` / `.bak` file persistence, `ContentAddressedStorage` routing,
//! generic `KeyValueStorage` (e.g., `eframe::Storage` adapters or in-memory maps), and
//! browser `localStorage` / `IndexedDB` / `Cache API` helpers on `wasm32`.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
#[cfg(not(target_arch = "wasm32"))]
use spodeian_cache::ContentAddressedStorage;
use spodeian_cache::PreferentialRouter;
#[allow(unused_imports)]
use spodeian_cache::StorageTier;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
#[allow(unused_imports)]
use tracing::{error, info, warn};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

pub use spodeian_web_utils::{
    trigger_binary_download, trigger_text_download, try_trigger_binary_download,
    try_trigger_text_download,
};

/// Errors arising during state persistence, deserialization, or multi-tiered routing.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum StorageError {
    #[error("Storage content is empty")]
    EmptyContent,

    #[error("Failed to deserialize state: {0}")]
    DeserializationFailed(String),

    #[error("All browser storage tiers failed")]
    AllTiersFailed,

    #[error("Web storage quota exceeded or access restricted: {0}")]
    QuotaExceeded(String),

    #[error("Browser window localStorage is unavailable in this context")]
    LocalStorageUnavailable,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cache error: {0}")]
    Cache(#[from] spodeian_cache::CacheError),
}

/// Active or target storage tier used to persist application state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StorageBackend {
    #[default]
    LocalStorage,
    IndexedDb,
    CacheApi,
    NativeCas,
    DiskFile,
    MemoryOnly,
}

impl StorageBackend {
    /// Short human-facing icon for status displays.
    pub fn icon(self) -> &'static str {
        match self {
            Self::LocalStorage => "⚡",
            Self::IndexedDb => "🗄️",
            Self::CacheApi => "📦",
            Self::NativeCas => "🔒",
            Self::DiskFile => "💾",
            Self::MemoryOnly => "💭",
        }
    }

    /// Descriptive label for diagnostics panels and logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::LocalStorage => "Local Storage (Fast Tier)",
            Self::IndexedDb => "IndexedDB (Structured Relational Tier)",
            Self::CacheApi => "Cache API (Large Binary Weights Tier)",
            Self::NativeCas => "Native CAS (Content-Addressed Disk Tier)",
            Self::DiskFile => "Local File System",
            Self::MemoryOnly => "In-Memory Only (Ephemeral)",
        }
    }

    /// Combines [`StorageBackend::icon`] and [`StorageBackend::label`].
    pub fn label_with_icon(self) -> String {
        format!("{} {}", self.icon(), self.label())
    }
}

/// Target working capacity of items held directly in RAM before spilling to durable storage.
pub const MEMORY_WORKING_SET_LIMIT: usize = 100;

/// Snapshot of runtime persistence capabilities, active tier, and PWA install status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageDiagnostics {
    pub is_persisted: Option<bool>,
    pub pwa_install_available: bool,
    pub is_pwa_installed: bool,
    pub backend: StorageBackend,
    pub quota_exceeded: bool,
    pub idb_active: bool,
    pub usage_bytes: u64,
    pub quota_bytes: u64,
    pub memory_items_count: usize,
    pub memory_capacity_limit: usize,
    pub fallback_tier_active: bool,
}

impl Default for StorageDiagnostics {
    fn default() -> Self {
        Self {
            is_persisted: None,
            pwa_install_available: false,
            is_pwa_installed: false,
            backend: StorageBackend::default(),
            quota_exceeded: false,
            idb_active: false,
            usage_bytes: 0,
            quota_bytes: 0,
            memory_items_count: 0,
            memory_capacity_limit: MEMORY_WORKING_SET_LIMIT,
            fallback_tier_active: false,
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn invoke_window_fn(name: &str, args: &[wasm_bindgen::JsValue]) -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let Ok(func_val) = js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str(name))
    else {
        return false;
    };
    let Some(func) = func_val.dyn_ref::<js_sys::Function>() else {
        return false;
    };
    match args {
        [] => func.call0(&window).is_ok(),
        [a0] => func.call1(&window, a0).is_ok(),
        [a0, a1] => func.call2(&window, a0, a1).is_ok(),
        _ => false,
    }
}

/// Queries current storage persistence and PWA status from the runtime environment.
#[allow(unused_mut)]
pub fn query_storage_diagnostics() -> StorageDiagnostics {
    let mut diag = StorageDiagnostics::default();

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(val) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__pwaInstallAvailable"),
            ) {
                diag.pwa_install_available = val.as_bool().unwrap_or(false);
            }
            if let Ok(val) =
                js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__pwaInstalled"))
            {
                diag.is_pwa_installed = val.as_bool().unwrap_or(false);
            }
            if let Ok(val) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__storagePersisted"),
            ) && let Some(b) = val.as_bool()
            {
                diag.is_persisted = Some(b);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        diag.backend = StorageBackend::DiskFile;
        diag.is_persisted = Some(true);
    }

    diag
}

/// Requests durable browser storage permission via window hook (no-op on native).
pub fn request_persistent_storage() {
    #[cfg(target_arch = "wasm32")]
    if invoke_window_fn("__requestPersistentStorage", &[]) {
        info!("Triggered __requestPersistentStorage from spodeian-persist");
    }
}

/// Triggers the browser PWA installation prompt via window hook (no-op on native).
pub fn trigger_pwa_install() {
    #[cfg(target_arch = "wasm32")]
    if invoke_window_fn("__triggerPWAInstall", &[]) {
        info!("Triggered __triggerPWAInstall from spodeian-persist");
    }
}

/// Controls the browser Screen Wake Lock hook on `wasm32` (no-op on native).
pub fn set_screen_wake_lock(active: bool) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = invoke_window_fn("__setWakeLock", &[wasm_bindgen::JsValue::from_bool(active)]);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = active;
    }
}

/// Schema-versioned persistence envelope wrapping serialized application state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VersionedSnapshot<T> {
    pub schema_version: u32,
    pub timestamp: f64,
    pub payload: T,
}

impl<T> VersionedSnapshot<T> {
    /// Wraps `payload` with an explicit schema version and UNIX timestamp.
    pub fn new(schema_version: u32, timestamp: f64, payload: T) -> Self {
        Self {
            schema_version,
            timestamp,
            payload,
        }
    }
}

/// Helper for encoding versioned JSON snapshots and decoding both versioned and raw payloads.
pub struct StateMigrator;

impl StateMigrator {
    /// Serializes `state` into a pretty-printed [`VersionedSnapshot`] JSON string.
    pub fn serialize_versioned<T: Serialize>(
        state: &T,
        schema_version: u32,
        timestamp: f64,
    ) -> Result<String, StorageError> {
        let snapshot = VersionedSnapshot::new(schema_version, timestamp, state);
        serde_json::to_string_pretty(&snapshot)
            .map_err(|e| StorageError::DeserializationFailed(e.to_string()))
    }

    /// Decodes either a [`VersionedSnapshot`] JSON envelope or a legacy raw `T` JSON payload.
    pub fn deserialize_versioned_or_raw<T: DeserializeOwned>(
        content: &str,
    ) -> Result<T, StorageError> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(StorageError::EmptyContent);
        }

        if let Ok(snapshot) = serde_json::from_str::<VersionedSnapshot<T>>(trimmed) {
            return Ok(snapshot.payload);
        }

        serde_json::from_str::<T>(trimmed)
            .map_err(|e| StorageError::DeserializationFailed(format!("JSON error: {e}")))
    }

    /// Decodes a versioned/raw JSON payload first, falling back to a caller-supplied parser
    /// (for example RON or custom legacy encodings).
    pub fn deserialize_with_fallback<T, F>(content: &str, fallback: F) -> Result<T, StorageError>
    where
        T: DeserializeOwned,
        F: FnOnce(&str) -> Result<T, String>,
    {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(StorageError::EmptyContent);
        }

        match Self::deserialize_versioned_or_raw::<T>(trimmed) {
            Ok(state) => Ok(state),
            Err(json_err) => fallback(trimmed).map_err(|fb_err| {
                StorageError::DeserializationFailed(format!("{json_err}; fallback error: {fb_err}"))
            }),
        }
    }
}

/// Abstract key-value persistence store so callers can adapt `eframe::Storage` or memory maps
/// without coupling `spodeian-persist` to a GUI framework.
pub trait KeyValueStorage {
    /// Reads a UTF-8 string value stored under `key`.
    fn get_string(&self, key: &str) -> Option<String>;

    /// Stores a UTF-8 string value under `key`.
    fn set_string(&mut self, key: &str, value: String);
}

impl KeyValueStorage for HashMap<String, String> {
    fn get_string(&self, key: &str) -> Option<String> {
        self.get(key).cloned()
    }

    fn set_string(&mut self, key: &str, value: String) {
        self.insert(key.to_string(), value);
    }
}

/// Restores serialized state by walking ordered storage keys across browser `localStorage`
/// (`wasm32`), an optional [`KeyValueStorage`], and an optional native disk file (with `.bak`
/// recovery).
pub fn load_state_multi_tier_keys<T: DeserializeOwned>(
    storage: Option<&dyn KeyValueStorage>,
    keys: &[&str],
    disk_fallback_path: Option<&Path>,
) -> Option<T> {
    #[cfg(target_arch = "wasm32")]
    if let Some(window) = web_sys::window()
        && let Ok(Some(local_storage)) = window.local_storage()
    {
        for &key in keys {
            if let Ok(Some(content)) = local_storage.get_item(key) {
                match StateMigrator::deserialize_versioned_or_raw::<T>(&content) {
                    Ok(state) => {
                        info!("Restored state from localStorage [{key}]");
                        return Some(state);
                    }
                    Err(e) => {
                        warn!("Failed to parse state from localStorage [{key}]: {e}");
                    }
                }
            }
        }
    }

    if let Some(kv) = storage {
        for &key in keys {
            if let Some(raw) = kv.get_string(key) {
                match StateMigrator::deserialize_versioned_or_raw::<T>(&raw) {
                    Ok(state) => {
                        info!("Restored state from KeyValueStorage [{key}]");
                        return Some(state);
                    }
                    Err(e) => {
                        warn!("Failed to parse state from KeyValueStorage [{key}]: {e}");
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = disk_fallback_path
        && let Some(state) = load_state_atomic_file::<T>(path)
    {
        return Some(state);
    }

    #[cfg(target_arch = "wasm32")]
    let _ = disk_fallback_path;

    None
}

/// Persists raw serialized JSON using preferential multi-tier routing:
/// - `wasm32`: Cache API (large payloads > 512 KB), `localStorage` (< 16 KB), IndexedDB fallback.
/// - native: content-addressed cache under `<temp_dir>/<cas_namespace>` plus optional atomic disk write.
pub fn save_state_multi_tier(
    key: &str,
    json_str: &str,
    cas_namespace: &str,
    disk_fallback_path: Option<&Path>,
) -> Result<StorageBackend, StorageError> {
    let size = json_str.len();
    let is_large_or_binary = size > 512 * 1024;
    let recommended_tier =
        PreferentialRouter::determine_tier(size, "application/json", is_large_or_binary);

    #[cfg(target_arch = "wasm32")]
    {
        let _ = (cas_namespace, disk_fallback_path);
        if let Some(window) = web_sys::window() {
            let k = wasm_bindgen::JsValue::from_str(key);
            let v = wasm_bindgen::JsValue::from_str(json_str);
            if recommended_tier == StorageTier::CacheApi
                && invoke_window_fn("__saveToCacheApi", &[k.clone(), v.clone()])
            {
                info!("Stored large payload in Cache API [{key}]");
                return Ok(StorageBackend::CacheApi);
            }

            let mut local_storage_saved = false;
            if size < 16 * 1024
                && let Ok(Some(storage)) = window.local_storage()
                && storage.set_item(key, json_str).is_ok()
            {
                local_storage_saved = true;
            }

            if invoke_window_fn("__saveToIndexedDB", &[k, v]) {
                info!("Synchronized payload to IndexedDB [{key}]");
                if !local_storage_saved {
                    return Ok(StorageBackend::IndexedDb);
                }
            }

            if local_storage_saved {
                return Ok(StorageBackend::LocalStorage);
            }

            return Err(StorageError::AllTiersFailed);
        }
        Ok(StorageBackend::MemoryOnly)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (key, &recommended_tier);
        let safe_ns = if cas_namespace.trim().is_empty() {
            "spodeian_persist_cas"
        } else {
            cas_namespace
        };
        let cas_dir = std::env::temp_dir().join(safe_ns);
        let cas_ok = ContentAddressedStorage::new(&cas_dir)
            .and_then(|cas| cas.put(json_str.as_bytes()))
            .is_ok();

        if let Some(path) = disk_fallback_path {
            write_atomic_str(path, json_str)?;
            return Ok(StorageBackend::DiskFile);
        }

        if cas_ok {
            return Ok(StorageBackend::NativeCas);
        }

        Ok(StorageBackend::MemoryOnly)
    }
}

fn companion_path(path: &Path, suffix: &str) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

fn write_atomic_str(path: &Path, content: &str) -> Result<(), StorageError> {
    use std::io::Write;

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let tmp_path = companion_path(path, ".tmp");
    let bak_path = companion_path(path, ".bak");

    {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(content.as_bytes())?;
        file.flush()?;
        file.sync_all()?;
    }

    if path.exists() {
        let _ = std::fs::copy(path, &bak_path);
    }

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Atomically writes `state` as a [`VersionedSnapshot`] JSON file: writes to `.tmp`, flushes
/// and syncs to disk, copies any existing primary file to `.bak`, and renames `.tmp` into place.
pub fn save_state_atomic_file<T: Serialize>(
    path: impl AsRef<Path>,
    state: &T,
    schema_version: u32,
    timestamp: f64,
) -> Result<(), StorageError> {
    let json = StateMigrator::serialize_versioned(state, schema_version, timestamp)?;
    write_atomic_str(path.as_ref(), &json)
}

/// Loads state from `path`, falling back to `<path>.bak` if the primary file is missing or corrupt.
pub fn load_state_atomic_file<T: DeserializeOwned>(path: impl AsRef<Path>) -> Option<T> {
    let path = path.as_ref();
    let bak_path = companion_path(path, ".bak");

    if let Ok(content) = std::fs::read_to_string(path) {
        match StateMigrator::deserialize_versioned_or_raw::<T>(&content) {
            Ok(state) => {
                info!("Restored state from primary file {}", path.display());
                return Some(state);
            }
            Err(e) => {
                warn!(
                    "Corrupt primary save file {}, checking backup {}: {e}",
                    path.display(),
                    bak_path.display()
                );
            }
        }
    }

    if let Ok(content) = std::fs::read_to_string(&bak_path) {
        match StateMigrator::deserialize_versioned_or_raw::<T>(&content) {
            Ok(state) => {
                info!("Restored state from backup file {}", bak_path.display());
                return Some(state);
            }
            Err(e) => {
                warn!("Backup save file {} is also corrupt: {e}", bak_path.display());
            }
        }
    }

    None
}
