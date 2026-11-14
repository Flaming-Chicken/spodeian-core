#![cfg(not(target_arch = "wasm32"))]

use serde::{Deserialize, Serialize};
use spodeian_persist::{
    KeyValueStorage, StateMigrator, StorageBackend, StorageError, VersionedSnapshot,
    load_state_atomic_file, load_state_multi_tier_keys, save_state_atomic_file,
    save_state_multi_tier,
};
use std::collections::HashMap;
use tempfile::tempdir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SampleState {
    name: String,
    count: u32,
}

#[test]
fn test_versioned_snapshot_roundtrip_and_raw_fallback() {
    let original = SampleState {
        name: "spodeian".into(),
        count: 42,
    };

    let encoded = StateMigrator::serialize_versioned(&original, 3, 1234.5).unwrap();
    let envelope: VersionedSnapshot<SampleState> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(envelope.schema_version, 3);
    assert_eq!(envelope.timestamp, 1234.5);
    assert_eq!(envelope.payload, original);

    let decoded: SampleState = StateMigrator::deserialize_versioned_or_raw(&encoded).unwrap();
    assert_eq!(decoded, original);

    let legacy_raw = r#"{"name":"legacy","count":7}"#;
    let decoded_legacy: SampleState =
        StateMigrator::deserialize_versioned_or_raw(legacy_raw).unwrap();
    assert_eq!(
        decoded_legacy,
        SampleState {
            name: "legacy".into(),
            count: 7,
        }
    );

    let custom: SampleState = StateMigrator::deserialize_with_fallback("alt:99", |raw| {
        let count = raw
            .strip_prefix("alt:")
            .ok_or_else(|| "missing prefix".to_string())?
            .parse::<u32>()
            .map_err(|e| e.to_string())?;
        Ok(SampleState {
            name: "alt".into(),
            count,
        })
    })
    .unwrap();
    assert_eq!(
        custom,
        SampleState {
            name: "alt".into(),
            count: 99,
        }
    );

    assert!(matches!(
        StateMigrator::deserialize_versioned_or_raw::<SampleState>("   "),
        Err(StorageError::EmptyContent)
    ));
}

#[test]
fn test_atomic_file_save_and_corrupt_primary_backup_recovery() {
    let dir = tempdir().unwrap();
    let save_path = dir.path().join("app.state.json");
    let bak_path = dir.path().join("app.state.json.bak");

    let first = SampleState {
        name: "first".into(),
        count: 1,
    };
    save_state_atomic_file(&save_path, &first, 1, 10.0).unwrap();
    assert!(save_path.exists());
    assert!(!bak_path.exists());

    let second = SampleState {
        name: "second".into(),
        count: 2,
    };
    save_state_atomic_file(&save_path, &second, 1, 20.0).unwrap();
    assert!(bak_path.exists());
    assert_eq!(load_state_atomic_file::<SampleState>(&save_path), Some(second));

    // Corrupt primary save and verify .bak (first) is restored.
    std::fs::write(&save_path, "{corrupt json").unwrap();
    assert_eq!(load_state_atomic_file::<SampleState>(&save_path), Some(first));
}

#[test]
fn test_key_value_multi_tier_priority_and_native_fallback() {
    let dir = tempdir().unwrap();
    let disk_path = dir.path().join("nested").join("session.json");

    let mut kv: HashMap<String, String> = HashMap::new();
    let primary_state = SampleState {
        name: "primary".into(),
        count: 10,
    };
    let legacy_state = SampleState {
        name: "legacy".into(),
        count: 5,
    };
    kv.set_string(
        "app_v2",
        StateMigrator::serialize_versioned(&primary_state, 2, 100.0).unwrap(),
    );
    kv.set_string("app_v1", serde_json::to_string(&legacy_state).unwrap());

    let loaded = load_state_multi_tier_keys::<SampleState>(
        Some(&kv),
        &["app_v2", "app_v1"],
        Some(&disk_path),
    );
    assert_eq!(loaded, Some(primary_state));

    let fallback_loaded = load_state_multi_tier_keys::<SampleState>(
        Some(&kv),
        &["missing_key", "app_v1"],
        Some(&disk_path),
    );
    assert_eq!(fallback_loaded, Some(legacy_state));

    // Verify native save_state_multi_tier routes to NativeCas when no disk path is supplied,
    // and to DiskFile when disk_fallback_path is supplied.
    let cas_backend = save_state_multi_tier(
        "cas_only",
        r#"{"name":"cas","count":11}"#,
        "spodeian_persist_test",
        None,
    )
    .unwrap();
    assert_eq!(cas_backend, StorageBackend::NativeCas);

    let disk_payload = StateMigrator::serialize_versioned(
        &SampleState {
            name: "disk".into(),
            count: 22,
        },
        2,
        200.0,
    )
    .unwrap();
    let disk_backend = save_state_multi_tier(
        "disk_key",
        &disk_payload,
        "spodeian_persist_test",
        Some(&disk_path),
    )
    .unwrap();
    assert_eq!(disk_backend, StorageBackend::DiskFile);

    let restored_from_disk =
        load_state_multi_tier_keys::<SampleState>(None, &["disk_key"], Some(&disk_path));
    assert_eq!(
        restored_from_disk,
        Some(SampleState {
            name: "disk".into(),
            count: 22,
        })
    );
}
