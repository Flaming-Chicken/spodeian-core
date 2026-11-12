use serde::{Deserialize, Serialize};
use spodeian_cache::{CacheError, ContentAddressedStorage, PreferentialRouter, StorageTier};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct TestModelWeights {
    layer_name: String,
    dimensions: (usize, usize),
    weights: Vec<i32>,
}

#[test]
fn test_cas_put_get_and_tamper_detection() {
    let temp_dir = std::env::temp_dir().join("spodeian_test_cas_tamper");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let cas = ContentAddressedStorage::new(&temp_dir).expect("create CAS");

    let original_data = b"Mamba-2-Ternary-Weights-Chunk-001";
    let hash = cas.put(original_data).expect("put data");

    assert!(cas.contains(&hash));
    let retrieved = cas.get(&hash).expect("get data");
    assert_eq!(retrieved, original_data);

    // Tamper with the file on disk to simulate disk corruption or MITM modification
    let file_path = temp_dir.join(&hash);
    std::fs::write(&file_path, b"Tampered-Corrupted-Data").expect("write tamper");

    let result = cas.get(&hash);
    match result {
        Err(CacheError::TamperDetected {
            key,
            expected,
            computed,
        }) => {
            assert_eq!(key, hash);
            assert_eq!(expected, hash);
            assert_ne!(computed, expected);
        }
        other => panic!("Expected TamperDetected, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_cas_json_entity_roundtrip() {
    let temp_dir = std::env::temp_dir().join("spodeian_test_cas_json");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let cas = ContentAddressedStorage::new(&temp_dir).expect("create CAS");

    let weights = TestModelWeights {
        layer_name: "ssm_in_proj".to_string(),
        dimensions: (64, 128),
        weights: vec![1, -1, 0, 1, 0, -1],
    };

    let hash = cas.put_json(&weights).expect("put json");
    let restored: TestModelWeights = cas.get_json(&hash).expect("get json");
    assert_eq!(restored, weights);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_preferential_router_tier_decision() {
    let native_tier =
        PreferentialRouter::determine_tier(1024 * 1024, "application/octet-stream", true);
    assert_eq!(native_tier, StorageTier::NativeCas);

    let small_native = PreferentialRouter::determine_tier(512, "application/json", false);
    assert_eq!(small_native, StorageTier::NativeCas);
}

#[test]
fn test_bounded_memory_cache_lru_eviction() {
    use spodeian_cache::BoundedMemoryCache;

    let mut cache = BoundedMemoryCache::new(3);
    assert_eq!(cache.capacity(), 3);
    assert!(cache.is_empty());

    assert_eq!(cache.put("k1", "v1"), None);
    assert_eq!(cache.put("k2", "v2"), None);
    assert_eq!(cache.put("k3", "v3"), None);
    assert_eq!(cache.len(), 3);

    // Access k1 so k2 becomes the LRU
    assert_eq!(cache.get(&"k1"), Some(&"v1"));

    // Inserting k4 should evict k2 (LRU)
    let evicted = cache.put("k4", "v4");
    assert_eq!(evicted, Some(("k2", "v2")));

    assert_eq!(cache.len(), 3);
    assert!(cache.contains_key(&"k1"));
    assert!(!cache.contains_key(&"k2"));
    assert!(cache.contains_key(&"k3"));
    assert!(cache.contains_key(&"k4"));
}

#[test]
fn test_tiered_memory_store_fallback_and_memory_protection() {
    use spodeian_cache::TieredMemoryStore;

    let temp_dir = std::env::temp_dir().join("spodeian_test_tiered_store");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Max 2 items in RAM; excess must spill to persistent CAS fallback
    let mut store = TieredMemoryStore::<String>::new(2, Some(&temp_dir)).expect("create store");
    assert_eq!(store.memory_capacity(), 2);

    store.put("item1", "value1".to_string()).expect("put item1");
    store.put("item2", "value2".to_string()).expect("put item2");

    let metrics = store.metrics();
    assert_eq!(metrics.memory_items_count, 2);
    assert_eq!(metrics.fallback_items_count, 0);

    // Putting item3 should evict item1 to fallback to prevent memory from overfilling!
    store.put("item3", "value3".to_string()).expect("put item3");

    let metrics = store.metrics();
    assert_eq!(metrics.memory_items_count, 2);
    assert_eq!(metrics.fallback_items_count, 1);
    assert_eq!(metrics.evictions_to_fallback, 1);

    // item3 is in memory (hit)
    assert_eq!(
        store.get("item3").expect("get item3"),
        Some("value3".to_string())
    );
    assert_eq!(store.metrics().memory_hits, 1);

    // item1 was evicted to fallback; querying it should succeed via fallback hit!
    assert_eq!(
        store.get("item1").expect("get item1"),
        Some("value1".to_string())
    );
    assert_eq!(store.metrics().fallback_hits, 1);

    // Memory capacity constraint was strictly preserved throughout
    assert!(store.metrics().memory_items_count <= 2);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

fn fresh_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("spodeian_test_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn test_cas_named_refs_survive_reopen() {
    use spodeian_cache::{CacheError, ContentAddressedStorage};

    let dir = fresh_dir("named_refs");
    {
        let cas = ContentAddressedStorage::new(&dir).expect("create cas");
        cas.put_named("app/state", b"version one").expect("put v1");
        cas.put_named("app/state", b"version two")
            .expect("overwrite");
        cas.put_named_json("app/prefs", &vec![1, 2, 3])
            .expect("put json");
    }

    let cas = ContentAddressedStorage::new(&dir).expect("reopen cas");
    assert_eq!(cas.get_named("app/state").expect("get"), b"version two");
    assert_eq!(
        cas.get_named_json::<Vec<i32>>("app/prefs")
            .expect("get json"),
        vec![1, 2, 3]
    );

    let mut names = cas.list_names().expect("list");
    names.sort();
    assert_eq!(
        names,
        vec!["app/prefs".to_string(), "app/state".to_string()]
    );

    assert!(cas.remove_named("app/state").expect("remove"));
    assert!(!cas.remove_named("app/state").expect("remove again"));
    assert!(matches!(
        cas.get_named("app/state"),
        Err(CacheError::NotFound(_))
    ));
    assert!(cas.resolve_name("never-stored").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_cas_list_names_on_empty_store() {
    use spodeian_cache::ContentAddressedStorage;

    let dir = fresh_dir("empty_names");
    let cas = ContentAddressedStorage::new(&dir).expect("create cas");
    assert!(cas.list_names().expect("list").is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tiered_memory_store_spilled_items_survive_reopen() {
    use spodeian_cache::TieredMemoryStore;

    let dir = fresh_dir("tiered_reopen");
    {
        let mut store = TieredMemoryStore::<String>::new(1, Some(&dir)).expect("create");
        store.put("a", "alpha".to_string()).expect("put a");
        store.put("b", "beta".to_string()).expect("put b; spills a");
        assert_eq!(store.metrics().fallback_items_count, 1);
    }

    let mut store = TieredMemoryStore::<String>::new(1, Some(&dir)).expect("reopen");
    assert_eq!(store.metrics().fallback_items_count, 1);
    assert!(store.contains("a"));
    assert_eq!(store.get("a").expect("get a"), Some("alpha".to_string()));
    // "b" was only in memory and was never flushed, so it did not survive.
    assert_eq!(store.get("b").expect("get b"), None);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tiered_memory_store_flush_persists_memory_tier() {
    use spodeian_cache::TieredMemoryStore;

    let dir = fresh_dir("tiered_flush");
    {
        let mut store = TieredMemoryStore::<String>::new(4, Some(&dir)).expect("create");
        store.put("x", "ex".to_string()).expect("put x");
        store.put("y", "why".to_string()).expect("put y");
        assert_eq!(store.flush().expect("flush"), 2);
        // Removing a key also removes its persisted reference.
        assert!(store.remove("y"));
    }

    let mut store = TieredMemoryStore::<String>::new(4, Some(&dir)).expect("reopen");
    assert_eq!(store.get("x").expect("get x"), Some("ex".to_string()));
    assert_eq!(store.get("y").expect("get y"), None);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tiered_memory_store_overwrite_replaces_spilled_value() {
    use spodeian_cache::TieredMemoryStore;

    let dir = fresh_dir("tiered_overwrite");
    {
        let mut store = TieredMemoryStore::<String>::new(1, Some(&dir)).expect("create");
        store.put("k", "old".to_string()).expect("put old");
        store.put("other", "o".to_string()).expect("spill k");
        store
            .put("k", "new".to_string())
            .expect("overwrite k; spills other");
        store.flush().expect("flush");
    }

    let mut store = TieredMemoryStore::<String>::new(1, Some(&dir)).expect("reopen");
    assert_eq!(store.get("k").expect("get k"), Some("new".to_string()));
    assert_eq!(
        store.get("other").expect("get other"),
        Some("o".to_string())
    );

    let _ = std::fs::remove_dir_all(&dir);
}
