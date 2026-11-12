//! Native tests for the asynchronous stores. Futures are driven by `pollster`.
#![cfg(not(target_arch = "wasm32"))]

use pollster::block_on;
use spodeian_cache::{AsyncTieredStore, PreferentialStore, StorageTier};

fn fresh_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("spodeian_async_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn async_tiered_store_is_write_through_and_bounded() {
    let dir = fresh_dir("write_through");
    block_on(async {
        let store = AsyncTieredStore::<u32>::with_disk(100, &dir).expect("open");
        for i in 0..101u32 {
            store.put(format!("item-{i:03}"), i).await.expect("put");
        }
        let metrics = store.metrics();
        assert_eq!(metrics.memory_items_count, 100);
        assert_eq!(metrics.fallback_items_count, 101);
        assert_eq!(metrics.evictions_to_fallback, 1);
        assert_eq!(store.len(), 101);

        // The first item was evicted from memory but is still served from disk.
        assert_eq!(store.get("item-000").await.expect("get"), Some(0));
        assert_eq!(store.metrics().fallback_hits, 1);
        assert_eq!(store.get("item-000").await.expect("get again"), Some(0));
        assert_eq!(store.metrics().memory_hits, 1);
    });
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn async_tiered_store_survives_reopen_and_removes() {
    let dir = fresh_dir("reopen");
    block_on(async {
        let store = AsyncTieredStore::<String>::with_disk(1, &dir).expect("open");
        store.put("a", "alpha".into()).await.expect("put a");
        store.put("b", "beta".into()).await.expect("put b");
        assert!(store.remove("a").await.expect("remove a"));
        assert!(!store.remove("a").await.expect("remove a again"));
        drop(store);

        let store = AsyncTieredStore::<String>::with_disk(1, &dir).expect("reopen");
        assert_eq!(store.keys(), vec!["b".to_string()]);
        assert_eq!(store.get("b").await.expect("get b"), Some("beta".into()));
        assert_eq!(store.get("a").await.expect("get a"), None);
        assert!(!store.contains("a"));
    });
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn async_tiered_store_in_memory_drops_evicted_items() {
    block_on(async {
        let store = AsyncTieredStore::<u8>::in_memory(1);
        store.put("x", 1).await.expect("put x");
        store.put("y", 2).await.expect("put y");
        assert_eq!(store.get("x").await.expect("get x"), None);
        assert_eq!(store.get("y").await.expect("get y"), Some(2));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    });
}

#[test]
fn preferential_store_native_round_trip() {
    let dir = fresh_dir("preferential");
    block_on(async {
        let store = PreferentialStore::open_native(&dir).expect("open");
        assert_eq!(
            store.save_json("state", &(1, "one")).await.expect("save"),
            StorageTier::NativeCas
        );
        store
            .save_bytes("blob", &[7u8; 1024], "application/octet-stream", true)
            .await
            .expect("save bytes");

        let reopened = PreferentialStore::open_native(&dir).expect("reopen");
        let state: Option<(u32, String)> = reopened.load_json("state").await.expect("load");
        assert_eq!(state, Some((1, "one".into())));
        assert_eq!(
            reopened.load_bytes("blob").await.expect("load bytes"),
            Some(vec![7u8; 1024])
        );

        assert!(reopened.remove("state").await.expect("remove"));
        assert_eq!(
            reopened.load_bytes("state").await.expect("load removed"),
            None
        );
        assert_eq!(
            reopened.load_bytes("never").await.expect("load missing"),
            None
        );
    });
    let _ = std::fs::remove_dir_all(&dir);
}
