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
