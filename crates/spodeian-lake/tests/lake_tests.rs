use spodeian_lake::{DataLakeConfig, TabularDataset, WindowsDataLake};

fn create_mock_market_dataset() -> TabularDataset {
    let mut dataset = TabularDataset::new();
    dataset
        .add_string_column("symbol", vec!["BHP.AX".to_string(), "BHP.AX".to_string(), "BHP.AX".to_string()])
        .unwrap();
    dataset
        .add_int_column("timestamp", vec![1700000000, 1700000060, 1700000120])
        .unwrap();
    dataset
        .add_float_column("close", vec![42.50, 42.75, 42.60])
        .unwrap();
    dataset
        .add_float_column("volume", vec![125000.0, 98000.0, 142000.0])
        .unwrap();
    dataset
}

#[test]
fn test_tabular_dataset_parquet_roundtrip() {
    let dataset = create_mock_market_dataset();

    let mut parquet_buffer = Vec::new();
    dataset.write_to_parquet(&mut parquet_buffer).expect("Parquet serialization failed");
    assert!(!parquet_buffer.is_empty());

    let restored = TabularDataset::read_from_parquet(&parquet_buffer).expect("Parquet deserialization failed");

    assert_eq!(restored.num_rows, 3);
    assert_eq!(restored.column_names, vec!["symbol", "timestamp", "close", "volume"]);
    assert_eq!(restored.string_columns.get("symbol").unwrap(), &vec!["BHP.AX", "BHP.AX", "BHP.AX"]);
    assert_eq!(restored.int_columns.get("timestamp").unwrap(), &vec![1700000000, 1700000060, 1700000120]);
    assert_eq!(restored.float_columns.get("close").unwrap(), &vec![42.50, 42.75, 42.60]);
    assert_eq!(restored.float_columns.get("volume").unwrap(), &vec![125000.0, 98000.0, 142000.0]);
}

#[test]
fn test_windows_data_lake_lifecycle() {
    let temp_dir = std::env::temp_dir().join(format!("spodeian_lake_test_{}", std::process::id()));
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: 1024 * 1024, // 1 MB
        default_ttl_seconds: Some(3600), // 1 hour
    };

    let mut lake = WindowsDataLake::new(config).expect("Failed to initialize data lake");
    let dataset = create_mock_market_dataset();

    // Store partition
    let current_epoch = 1700000000;
    let meta = lake
        .store_parquet("market_ticks", "asx_2026_09", &dataset, Some(3600), current_epoch)
        .expect("Failed to store parquet");

    assert_eq!(meta.dataset_id, "market_ticks");
    assert_eq!(meta.record_count, 3);
    assert!(meta.byte_size > 0);

    // Load partition
    let loaded = lake
        .load_parquet("market_ticks", "asx_2026_09", current_epoch + 10)
        .expect("Failed to load parquet");
    assert_eq!(loaded.num_rows, 3);
    assert_eq!(loaded.float_columns.get("close").unwrap(), &vec![42.50, 42.75, 42.60]);

    // Check summary
    let summary = lake.summary();
    assert_eq!(summary.total_partitions, 1);
    assert_eq!(summary.total_records, 3);

    // Record training feedback: model has trained and utility has saturated (dropped to 0.1)
    lake.record_training_feedback("market_ticks", "asx_2026_09", 0.1, None, 10).unwrap();

    // Test TTL staleness pruning (advance time past TTL: +7200s and utility <= 0.4)
    let evicted = lake.prune_stale(current_epoch + 7200).expect("Failed to prune");
    assert_eq!(evicted, 1);
    assert_eq!(lake.summary().total_partitions, 0);

    // Cleanup temp dir
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_windows_data_lake_lru_capacity_eviction() {
    let temp_dir = std::env::temp_dir().join(format!("spodeian_lru_test_{}", std::process::id()));
    
    // Create first partition to gauge size
    let dataset = create_mock_market_dataset();
    let mut sample_buf = Vec::new();
    dataset.write_to_parquet(&mut sample_buf).unwrap();
    let single_partition_size = sample_buf.len() as u64;

    // Set capacity to hold only ~1 partition
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: single_partition_size + 200, // can fit 1 partition, not 2
        default_ttl_seconds: None,
    };

    let mut lake = WindowsDataLake::new(config).expect("Failed to initialize lake");

    // Store partition 1 at t=100
    lake.store_parquet("ticks", "part_1", &dataset, None, 100).unwrap();
    assert_eq!(lake.summary().total_partitions, 1);

    // Store partition 2 at t=200 -> partition 1 (older access) should be LRU evicted
    lake.store_parquet("ticks", "part_2", &dataset, None, 200).unwrap();

    let summary = lake.summary();
    assert_eq!(summary.total_partitions, 1);

    // Verify part_2 is present and part_1 was evicted
    assert!(lake.load_parquet("ticks", "part_2", 210).is_ok());
    assert!(lake.load_parquet("ticks", "part_1", 220).is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_value_aware_training_retention_and_pruning() {
    let temp_dir = std::env::temp_dir().join(format!("spodeian_value_test_{}", std::process::id()));
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: 10 * 1024 * 1024,
        default_ttl_seconds: None,
    };

    let mut lake = WindowsDataLake::new(config).expect("Failed to initialize lake");
    let dataset = create_mock_market_dataset();

    // 1. Ingest normal partition A (will later saturate)
    lake.store_parquet("market", "partition_a", &dataset, None, 1000).unwrap();

    // 2. Ingest crisis memory partition B (pinned permanently)
    lake.store_parquet_with_options("market", "crisis_2020", &dataset, None, 0.95, true, 1000).unwrap();

    // 3. Ingest high-loss active learning partition C
    lake.store_parquet("market", "partition_c", &dataset, None, 1000).unwrap();

    // Simulate training feedback:
    // Partition A: model has mastered it (utility drops to 0.05, seen 25 epochs)
    lake.record_training_feedback("market", "partition_a", 0.05, Some(0.0), 25).unwrap();

    // Partition C: model still has high reconstruction error (utility remains 0.90, seen only 2 epochs)
    lake.record_training_feedback("market", "partition_c", 0.90, Some(0.2), 2).unwrap();

    // Run utility-based pruning: threshold 0.15
    let evicted = lake.prune_low_utility(0.15).expect("Prune low utility failed");
    assert_eq!(evicted, 1, "Only partition A should be pruned as model won't benefit from it");

    // Verify partition A is gone, but crisis_2020 and partition_c are strictly preserved
    assert!(lake.load_parquet("market", "partition_a", 1010).is_err());
    assert!(lake.load_parquet("market", "crisis_2020", 1010).is_ok());
    assert!(lake.load_parquet("market", "partition_c", 1010).is_ok());

    let _ = std::fs::remove_dir_all(&temp_dir);
}
