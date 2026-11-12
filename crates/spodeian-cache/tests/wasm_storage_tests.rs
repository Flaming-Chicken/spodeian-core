//! Browser tests for the web storage tiers. Run with wasm-bindgen-test in a headless browser.
#![cfg(target_arch = "wasm32")]

use spodeian_cache::web::{
    CacheApiStore, IndexedDbStore, LocalStorageStore, is_storage_persisted, storage_estimate,
};
use spodeian_cache::{AsyncTieredStore, PreferentialStore, StorageTier};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn unique(name: &str) -> String {
    format!("{name}-{}", (js_sys::Math::random() * 1e12) as u64)
}

#[wasm_bindgen_test]
async fn indexed_db_round_trip_keys_and_reopen() {
    let db = unique("idb");
    let store = IndexedDbStore::open(&db, "records").await.expect("open");
    store.put_bytes("b", &[1, 2, 3]).await.expect("put b");
    store.put_json("a", &vec!["x"]).await.expect("put a");
    assert_eq!(
        store.get_bytes("b").await.expect("get"),
        Some(vec![1, 2, 3])
    );
    assert_eq!(store.get_bytes("missing").await.expect("get missing"), None);
    assert_eq!(store.keys().await.expect("keys"), vec!["a", "b"]);
    store.delete("b").await.expect("delete");
    store.delete("b").await.expect("delete missing is fine");
    store.close();

    let reopened = IndexedDbStore::open(&db, "records").await.expect("reopen");
    let a: Option<Vec<String>> = reopened.get_json("a").await.expect("get json");
    assert_eq!(a, Some(vec!["x".to_string()]));
    assert_eq!(reopened.keys().await.expect("keys"), vec!["a"]);
}

#[wasm_bindgen_test]
async fn indexed_db_adds_a_second_store_to_an_existing_database() {
    let db = unique("idb-two");
    let first = IndexedDbStore::open(&db, "one").await.expect("open one");
    first.put_bytes("k", &[1]).await.expect("put one");
    // Opening a new store name upgrades the database; the first handle is closed by it.
    let second = IndexedDbStore::open(&db, "two").await.expect("open two");
    second.put_bytes("k", &[2]).await.expect("put two");
    let first = IndexedDbStore::open(&db, "one").await.expect("reopen one");
    assert_eq!(first.get_bytes("k").await.expect("get one"), Some(vec![1]));
    assert_eq!(second.get_bytes("k").await.expect("get two"), Some(vec![2]));
}

#[wasm_bindgen_test]
async fn cache_api_round_trip() {
    let cache = CacheApiStore::open(&unique("cache")).await.expect("open");
    let blob = vec![9u8; 4096];
    cache
        .put_bytes("models/w a.bin", &blob, "application/octet-stream")
        .await
        .expect("put");
    assert_eq!(
        cache.get_bytes("models/w a.bin").await.expect("get"),
        Some(blob)
    );
    assert!(cache.delete("models/w a.bin").await.expect("delete"));
    assert_eq!(
        cache.get_bytes("models/w a.bin").await.expect("get gone"),
        None
    );
}

#[wasm_bindgen_test]
fn local_storage_round_trip() {
    let store = LocalStorageStore::open(&unique("ls")).expect("open");
    store.set("k", "v").expect("set");
    assert_eq!(store.get("k").expect("get"), Some("v".into()));
    store.remove("k").expect("remove");
    assert_eq!(store.get("k").expect("get gone"), None);
}

#[wasm_bindgen_test]
async fn preferential_store_routes_and_clears_stale_tiers() {
    let store = PreferentialStore::open_web(&unique("pref"))
        .await
        .expect("open");

    let tier = store
        .save_json("state", &vec![1, 2, 3])
        .await
        .expect("save json");
    assert_eq!(tier, StorageTier::IndexedDb);

    let tier = store
        .save_bytes("note", b"hello", "text/plain", false)
        .await
        .expect("save text");
    assert_eq!(tier, StorageTier::LocalStorage);

    let big = vec![5u8; 600 * 1024];
    let tier = store
        .save_bytes("state", &big, "application/octet-stream", false)
        .await
        .expect("overwrite state with binary");
    assert_eq!(tier, StorageTier::CacheApi);
    // The IndexedDB copy was removed, so the load sees the new value.
    assert_eq!(store.load_bytes("state").await.expect("load"), Some(big));

    let loaded: Option<Vec<u8>> = store.load_bytes("note").await.expect("load text");
    assert_eq!(loaded.as_deref(), Some(&b"hello"[..]));

    assert!(store.remove("state").await.expect("remove"));
    assert_eq!(store.load_bytes("state").await.expect("load removed"), None);
}

#[wasm_bindgen_test]
async fn async_tiered_store_spills_to_indexed_db_and_reloads() {
    let db = unique("tiered");
    let store = AsyncTieredStore::<u32>::with_indexed_db(
        2,
        IndexedDbStore::open(&db, "items").await.expect("open idb"),
    )
    .await
    .expect("open store");
    for i in 0..3u32 {
        store.put(format!("k{i}"), i).await.expect("put");
    }
    assert_eq!(store.metrics().memory_items_count, 2);
    assert_eq!(store.metrics().fallback_items_count, 3);
    assert_eq!(store.get("k0").await.expect("get evicted"), Some(0));
    drop(store);

    let reopened = AsyncTieredStore::<u32>::with_indexed_db(
        2,
        IndexedDbStore::open(&db, "items")
            .await
            .expect("reopen idb"),
    )
    .await
    .expect("reopen store");
    assert_eq!(reopened.keys(), vec!["k0", "k1", "k2"]);
    assert_eq!(reopened.get("k2").await.expect("get"), Some(2));
}

#[wasm_bindgen_test]
async fn storage_manager_helpers_resolve() {
    let estimate = storage_estimate().await.expect("estimate");
    assert!(estimate.quota_bytes > 0.0);
    let _ = is_storage_persisted().await.expect("persisted");
}
