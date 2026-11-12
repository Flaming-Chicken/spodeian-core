//! Browser storage backends for the tiers [`crate::PreferentialRouter`] chooses on the web.
//!
//! **Stability: Stable** — semver-guaranteed since v1.1. Available on `wasm32` only.
//!
//! - [`IndexedDbStore`]: durable, asynchronous key–value storage for structured records.
//! - [`CacheApiStore`]: the browser Cache API, for large or immutable binary assets.
//! - [`LocalStorageStore`]: small, synchronous string values.
//! - [`request_persistent_storage`], [`is_storage_persisted`] and [`storage_estimate`]
//!   wrap the `StorageManager` API.
//!
//! Every operation reports failures as [`CacheError::BrowserStorage`] rather than
//! panicking, so callers can tell a user their data was not saved.

use crate::CacheError;
use js_sys::{Array, Function, Promise, Uint8Array};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Event, IdbDatabase, IdbObjectStore, IdbOpenDbRequest, IdbRequest, IdbTransaction,
    IdbTransactionMode,
};

fn js_error(context: &str, value: JsValue) -> CacheError {
    let detail = value
        .dyn_ref::<web_sys::DomException>()
        .map(|e| format!("{}: {}", e.name(), e.message()))
        .or_else(|| {
            value
                .dyn_ref::<js_sys::Error>()
                .map(|e| String::from(e.message()))
        })
        .or_else(|| value.as_string())
        .unwrap_or_else(|| format!("{value:?}"));
    CacheError::BrowserStorage(format!("{context}: {detail}"))
}

fn window() -> Result<web_sys::Window, CacheError> {
    web_sys::window().ok_or_else(|| CacheError::BrowserStorage("no window available".into()))
}

/// Resolves when an event target fires one of the given success events, and rejects on
/// `error` or `abort`. The handler closure stays alive until the promise settles, and the
/// handlers are cleared afterwards, so nothing leaks.
async fn await_events(
    install: impl Fn(Option<&Function>),
    succeed_on: &'static [&'static str],
    result: impl Fn() -> JsValue + 'static,
    error: impl Fn() -> JsValue + 'static,
) -> Result<JsValue, JsValue> {
    let settle: Rc<RefCell<Option<(Function, Function)>>> = Rc::new(RefCell::new(None));
    let promise = {
        let settle = settle.clone();
        Promise::new(&mut move |resolve, reject| {
            *settle.borrow_mut() = Some((resolve, reject));
        })
    };
    let handler = {
        let settle = settle.clone();
        Closure::<dyn FnMut(Event)>::new(move |event: Event| {
            if let Some((resolve, reject)) = settle.borrow_mut().take() {
                if succeed_on.contains(&event.type_().as_str()) {
                    let _ = resolve.call1(&JsValue::NULL, &result());
                } else {
                    let _ = reject.call1(&JsValue::NULL, &error());
                }
            }
        })
    };
    install(Some(handler.as_ref().unchecked_ref()));
    let outcome = JsFuture::from(promise).await;
    install(None);
    drop(handler);
    outcome
}

async fn await_request(request: &IdbRequest) -> Result<JsValue, JsValue> {
    let (r1, r2, r3) = (request.clone(), request.clone(), request.clone());
    await_events(
        move |h| {
            r1.set_onsuccess(h);
            r1.set_onerror(h);
        },
        &["success"],
        move || r2.result().unwrap_or(JsValue::UNDEFINED),
        move || {
            r3.error()
                .ok()
                .flatten()
                .map(JsValue::from)
                .unwrap_or_else(|| JsValue::from_str("request failed"))
        },
    )
    .await
}

async fn await_transaction(tx: &IdbTransaction) -> Result<(), JsValue> {
    let (t1, t2) = (tx.clone(), tx.clone());
    await_events(
        move |h| {
            t1.set_oncomplete(h);
            t1.set_onerror(h);
            t1.set_onabort(h);
        },
        &["complete"],
        || JsValue::UNDEFINED,
        move || {
            t2.error()
                .map(JsValue::from)
                .unwrap_or_else(|| JsValue::from_str("transaction aborted"))
        },
    )
    .await
    .map(|_| ())
}

// ---------------------------------------------------------------------------
// IndexedDB
// ---------------------------------------------------------------------------

/// An asynchronous key–value store backed by one IndexedDB object store.
///
/// Values are stored as bytes (`Uint8Array`). Writes resolve only once the transaction
/// has committed, so a successful `put` means the data is durable.
///
/// Several stores can share one database: opening a store name the database does not
/// have yet upgrades the database to add it. An upgrade closes every other open
/// connection to that database, including other `IndexedDbStore` handles, which must then
/// be reopened. Prefer one database per store where that matters.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
#[derive(Debug, Clone)]
pub struct IndexedDbStore {
    db: IdbDatabase,
    store_name: String,
}

impl IndexedDbStore {
    /// Opens (creating if needed) `store_name` in the database `db_name`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn open(db_name: &str, store_name: &str) -> Result<Self, CacheError> {
        let db = Self::open_db(db_name, None, store_name).await?;
        if db.object_store_names().contains(store_name) {
            return Ok(Self {
                db,
                store_name: store_name.to_string(),
            });
        }
        // The database exists without this store: reopen at the next version to add it.
        let next_version = db.version() as u32 + 1;
        db.close();
        let db = Self::open_db(db_name, Some(next_version), store_name).await?;
        Ok(Self {
            db,
            store_name: store_name.to_string(),
        })
    }

    async fn open_db(
        db_name: &str,
        version: Option<u32>,
        store_name: &str,
    ) -> Result<IdbDatabase, CacheError> {
        let factory = window()?
            .indexed_db()
            .map_err(|e| js_error("indexedDB unavailable", e))?
            .ok_or_else(|| CacheError::BrowserStorage("indexedDB unavailable".into()))?;
        let request: IdbOpenDbRequest = match version {
            Some(v) => factory.open_with_u32(db_name, v),
            None => factory.open(db_name),
        }
        .map_err(|e| js_error("opening IndexedDB", e))?;

        // Creates the object store during a version upgrade. Kept alive until the open settles.
        let upgrade = {
            let request = request.clone();
            let store_name = store_name.to_string();
            Closure::<dyn FnMut(Event)>::new(move |_event: Event| {
                if let Ok(result) = request.result()
                    && let Ok(db) = result.dyn_into::<IdbDatabase>()
                    && !db.object_store_names().contains(&store_name)
                {
                    let _ = db.create_object_store(&store_name);
                }
            })
        };
        request.set_onupgradeneeded(Some(upgrade.as_ref().unchecked_ref()));
        let opened = await_request(&request).await;
        request.set_onupgradeneeded(None);
        drop(upgrade);

        let db = opened
            .map_err(|e| js_error("opening IndexedDB", e))?
            .dyn_into::<IdbDatabase>()
            .map_err(|e| js_error("opening IndexedDB", e))?;

        // Close this connection if another connection (or tab) upgrades the database, so the
        // upgrade is never blocked. The closure is freed when it runs.
        let to_close = db.clone();
        let on_version_change = Closure::once_into_js(move || to_close.close());
        db.set_onversionchange(Some(on_version_change.unchecked_ref()));
        Ok(db)
    }

    fn object_store(
        &self,
        mode: IdbTransactionMode,
    ) -> Result<(IdbTransaction, IdbObjectStore), CacheError> {
        let tx = self
            .db
            .transaction_with_str_and_mode(&self.store_name, mode)
            .map_err(|e| js_error("starting transaction", e))?;
        let store = tx
            .object_store(&self.store_name)
            .map_err(|e| js_error("opening object store", e))?;
        Ok((tx, store))
    }

    /// Stores `bytes` under `key`, replacing any existing value.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn put_bytes(&self, key: &str, bytes: &[u8]) -> Result<(), CacheError> {
        let (tx, store) = self.object_store(IdbTransactionMode::Readwrite)?;
        let value = Uint8Array::from(bytes);
        // A failed request aborts its transaction, so waiting for the commit covers both.
        store
            .put_with_key(&value, &JsValue::from_str(key))
            .map_err(|e| js_error("writing to IndexedDB", e))?;
        await_transaction(&tx)
            .await
            .map_err(|e| js_error("committing IndexedDB write", e))
    }

    /// Retrieves the bytes stored under `key`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn get_bytes(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        let (_tx, store) = self.object_store(IdbTransactionMode::Readonly)?;
        let request = store
            .get(&JsValue::from_str(key))
            .map_err(|e| js_error("reading from IndexedDB", e))?;
        let value = await_request(&request)
            .await
            .map_err(|e| js_error("reading from IndexedDB", e))?;
        if value.is_undefined() || value.is_null() {
            return Ok(None);
        }
        let array = value
            .dyn_into::<Uint8Array>()
            .map_err(|_| CacheError::BrowserStorage(format!("value for '{key}' is not bytes")))?;
        Ok(Some(array.to_vec()))
    }

    /// Stores a serializable value as JSON under `key`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn put_json<T: serde::Serialize>(
        &self,
        key: &str,
        value: &T,
    ) -> Result<(), CacheError> {
        self.put_bytes(key, &serde_json::to_vec(value)?).await
    }

    /// Retrieves and deserializes the JSON value stored under `key`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<T>, CacheError> {
        match self.get_bytes(key).await? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Deletes the value stored under `key`. Deleting a missing key succeeds.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn delete(&self, key: &str) -> Result<(), CacheError> {
        let (tx, store) = self.object_store(IdbTransactionMode::Readwrite)?;
        store
            .delete(&JsValue::from_str(key))
            .map_err(|e| js_error("deleting from IndexedDB", e))?;
        await_transaction(&tx)
            .await
            .map_err(|e| js_error("committing IndexedDB delete", e))
    }

    /// Lists every key in the store, in IndexedDB key order.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn keys(&self) -> Result<Vec<String>, CacheError> {
        let (_tx, store) = self.object_store(IdbTransactionMode::Readonly)?;
        let request = store
            .get_all_keys()
            .map_err(|e| js_error("listing IndexedDB keys", e))?;
        let value = await_request(&request)
            .await
            .map_err(|e| js_error("listing IndexedDB keys", e))?;
        Ok(Array::from(&value)
            .iter()
            .filter_map(|k| k.as_string())
            .collect())
    }

    /// Closes the underlying database connection.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn close(&self) {
        self.db.close();
    }
}

// ---------------------------------------------------------------------------
// Cache API
// ---------------------------------------------------------------------------

/// Binary storage in one named browser Cache API cache.
///
/// Keys map to synthetic same-origin URLs under `/__spodeian_cache__/`, which are never
/// fetched from the network.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
#[derive(Debug, Clone)]
pub struct CacheApiStore {
    cache: web_sys::Cache,
}

impl CacheApiStore {
    /// Opens (creating if needed) the cache `cache_name`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn open(cache_name: &str) -> Result<Self, CacheError> {
        let caches = window()?
            .caches()
            .map_err(|e| js_error("Cache API unavailable", e))?;
        let cache = JsFuture::from(caches.open(cache_name))
            .await
            .map_err(|e| js_error("opening cache", e))?
            .dyn_into::<web_sys::Cache>()
            .map_err(|e| js_error("opening cache", e))?;
        Ok(Self { cache })
    }

    fn url(key: &str) -> String {
        format!(
            "/__spodeian_cache__/{}",
            String::from(js_sys::encode_uri_component(key))
        )
    }

    /// Stores `bytes` under `key` with the given MIME type.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn put_bytes(
        &self,
        key: &str,
        bytes: &[u8],
        content_type: &str,
    ) -> Result<(), CacheError> {
        let headers = web_sys::Headers::new().map_err(|e| js_error("building headers", e))?;
        headers
            .set("content-type", content_type)
            .map_err(|e| js_error("building headers", e))?;
        let init = web_sys::ResponseInit::new();
        init.set_headers(&headers);
        let mut body = bytes.to_vec();
        let response = web_sys::Response::new_with_opt_u8_array_and_init(Some(&mut body), &init)
            .map_err(|e| js_error("building response", e))?;
        JsFuture::from(self.cache.put_with_str(&Self::url(key), &response))
            .await
            .map_err(|e| js_error("writing to cache", e))?;
        Ok(())
    }

    /// Retrieves the bytes stored under `key`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn get_bytes(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        let matched = JsFuture::from(self.cache.match_with_str(&Self::url(key)))
            .await
            .map_err(|e| js_error("reading from cache", e))?;
        if matched.is_undefined() || matched.is_null() {
            return Ok(None);
        }
        let response = matched
            .dyn_into::<web_sys::Response>()
            .map_err(|e| js_error("reading from cache", e))?;
        let buffer = JsFuture::from(
            response
                .array_buffer()
                .map_err(|e| js_error("reading cached body", e))?,
        )
        .await
        .map_err(|e| js_error("reading cached body", e))?;
        Ok(Some(Uint8Array::new(&buffer).to_vec()))
    }

    /// Deletes the entry under `key`. Returns whether one existed.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub async fn delete(&self, key: &str) -> Result<bool, CacheError> {
        let deleted = JsFuture::from(self.cache.delete_with_str(&Self::url(key)))
            .await
            .map_err(|e| js_error("deleting from cache", e))?;
        Ok(deleted.as_bool().unwrap_or(false))
    }
}

// ---------------------------------------------------------------------------
// LocalStorage
// ---------------------------------------------------------------------------

/// Small synchronous string storage in `window.localStorage`, with every key namespaced
/// by a prefix so several stores can share the origin.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
#[derive(Debug, Clone)]
pub struct LocalStorageStore {
    storage: web_sys::Storage,
    prefix: String,
}

impl LocalStorageStore {
    /// Opens local storage, prefixing every key with `prefix`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn open(prefix: &str) -> Result<Self, CacheError> {
        let storage = window()?
            .local_storage()
            .map_err(|e| js_error("localStorage unavailable", e))?
            .ok_or_else(|| CacheError::BrowserStorage("localStorage unavailable".into()))?;
        Ok(Self {
            storage,
            prefix: prefix.to_string(),
        })
    }

    /// Stores `value` under `key`. Fails when the browser's quota is exceeded.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn set(&self, key: &str, value: &str) -> Result<(), CacheError> {
        self.storage
            .set_item(&format!("{}{key}", self.prefix), value)
            .map_err(|e| js_error("writing to localStorage", e))
    }

    /// Retrieves the value under `key`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn get(&self, key: &str) -> Result<Option<String>, CacheError> {
        self.storage
            .get_item(&format!("{}{key}", self.prefix))
            .map_err(|e| js_error("reading from localStorage", e))
    }

    /// Removes the value under `key`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
    pub fn remove(&self, key: &str) -> Result<(), CacheError> {
        self.storage
            .remove_item(&format!("{}{key}", self.prefix))
            .map_err(|e| js_error("removing from localStorage", e))
    }
}

// ---------------------------------------------------------------------------
// StorageManager
// ---------------------------------------------------------------------------

/// Storage usage and quota for this origin, in bytes.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StorageEstimate {
    /// Bytes currently used by this origin.
    pub usage_bytes: f64,
    /// Bytes this origin may use before writes fail.
    pub quota_bytes: f64,
}

fn storage_manager() -> Result<web_sys::StorageManager, CacheError> {
    Ok(window()?.navigator().storage())
}

/// Asks the browser to keep this origin's storage when space runs low. Browsers may grant
/// or refuse silently; returns whether storage is now persistent.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub async fn request_persistent_storage() -> Result<bool, CacheError> {
    let promise = storage_manager()?
        .persist()
        .map_err(|e| js_error("requesting persistent storage", e))?;
    let granted = JsFuture::from(promise)
        .await
        .map_err(|e| js_error("requesting persistent storage", e))?;
    Ok(granted.as_bool().unwrap_or(false))
}

/// Returns whether this origin's storage is currently persistent.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub async fn is_storage_persisted() -> Result<bool, CacheError> {
    let promise = storage_manager()?
        .persisted()
        .map_err(|e| js_error("checking storage persistence", e))?;
    let persisted = JsFuture::from(promise)
        .await
        .map_err(|e| js_error("checking storage persistence", e))?;
    Ok(persisted.as_bool().unwrap_or(false))
}

/// Returns this origin's storage usage and quota.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub async fn storage_estimate() -> Result<StorageEstimate, CacheError> {
    let promise = storage_manager()?
        .estimate()
        .map_err(|e| js_error("estimating storage", e))?;
    let estimate: web_sys::StorageEstimate = JsFuture::from(promise)
        .await
        .map_err(|e| js_error("estimating storage", e))?
        .unchecked_into();
    Ok(StorageEstimate {
        usage_bytes: estimate.get_usage().unwrap_or(0.0),
        quota_bytes: estimate.get_quota().unwrap_or(0.0),
    })
}
