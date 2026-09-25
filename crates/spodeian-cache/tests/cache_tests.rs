use spodeian_cache::{CacheError, ContentAddressedStorage, PreferentialRouter, StorageTier};
use serde::{Deserialize, Serialize};

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
        Err(CacheError::TamperDetected { key, expected, computed }) => {
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
    let native_tier = PreferentialRouter::determine_tier(1024 * 1024, "application/octet-stream", true);
    assert_eq!(native_tier, StorageTier::NativeCas);

    let small_native = PreferentialRouter::determine_tier(512, "application/json", false);
    assert_eq!(small_native, StorageTier::NativeCas);
}
