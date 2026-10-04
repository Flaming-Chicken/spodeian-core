use spodeian_lake::{DataLakeConfig, TabularDataset, WindowsDataLake};

fn create_mock_market_dataset() -> TabularDataset {
    let mut dataset = TabularDataset::new();
    dataset
        .add_string_column(
            "symbol",
            vec![
                "BHP.AX".to_string(),
                "BHP.AX".to_string(),
                "BHP.AX".to_string(),
            ],
        )
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
    dataset
        .write_to_parquet(&mut parquet_buffer)
        .expect("Parquet serialization failed");
    assert!(!parquet_buffer.is_empty());

    let restored =
        TabularDataset::read_from_parquet(&parquet_buffer).expect("Parquet deserialization failed");

    assert_eq!(restored.num_rows, 3);
    assert_eq!(
        restored.column_names,
        vec!["symbol", "timestamp", "close", "volume"]
    );
    assert_eq!(
        restored.string_columns.get("symbol").unwrap(),
        &vec!["BHP.AX", "BHP.AX", "BHP.AX"]
    );
    assert_eq!(
        restored.int_columns.get("timestamp").unwrap(),
        &vec![1700000000, 1700000060, 1700000120]
    );
    assert_eq!(
        restored.float_columns.get("close").unwrap(),
        &vec![42.50, 42.75, 42.60]
    );
    assert_eq!(
        restored.float_columns.get("volume").unwrap(),
        &vec![125000.0, 98000.0, 142000.0]
    );
}

#[test]
fn test_windows_data_lake_lifecycle() {
    let temp_dir = std::env::temp_dir().join(format!("spodeian_lake_test_{}", std::process::id()));
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: 1024 * 1024,  // 1 MB
        default_ttl_seconds: Some(3600), // 1 hour
    };

    let mut lake = WindowsDataLake::new(config).expect("Failed to initialize data lake");
    let dataset = create_mock_market_dataset();

    // Store partition
    let current_epoch = 1700000000;
    let meta = lake
        .store_parquet(
            "market_ticks",
            "asx_2026_09",
            &dataset,
            Some(3600),
            current_epoch,
        )
        .expect("Failed to store parquet");

    assert_eq!(meta.dataset_id, "market_ticks");
    assert_eq!(meta.record_count, 3);
    assert!(meta.byte_size > 0);

    // Load partition
    let loaded = lake
        .load_parquet("market_ticks", "asx_2026_09", current_epoch + 10)
        .expect("Failed to load parquet");
    assert_eq!(loaded.num_rows, 3);
    assert_eq!(
        loaded.float_columns.get("close").unwrap(),
        &vec![42.50, 42.75, 42.60]
    );

    // Check summary
    let summary = lake.summary();
    assert_eq!(summary.total_partitions, 1);
    assert_eq!(summary.total_records, 3);

    // Record training feedback: model has trained and utility has saturated (dropped to 0.1)
    lake.record_training_feedback("market_ticks", "asx_2026_09", 0.1, None, 10)
        .unwrap();

    // Test TTL staleness pruning (advance time past TTL: +7200s and utility <= 0.4)
    let evicted = lake
        .prune_stale(current_epoch + 7200)
        .expect("Failed to prune");
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
    lake.store_parquet("ticks", "part_1", &dataset, None, 100)
        .unwrap();
    assert_eq!(lake.summary().total_partitions, 1);

    // Store partition 2 at t=200 -> partition 1 (older access) should be LRU evicted
    lake.store_parquet("ticks", "part_2", &dataset, None, 200)
        .unwrap();

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
    lake.store_parquet("market", "partition_a", &dataset, None, 1000)
        .unwrap();

    // 2. Ingest crisis memory partition B (pinned permanently)
    lake.store_parquet_with_options("market", "crisis_2020", &dataset, None, 0.95, true, 1000)
        .unwrap();

    // 3. Ingest high-loss active learning partition C
    lake.store_parquet("market", "partition_c", &dataset, None, 1000)
        .unwrap();

    // Simulate training feedback:
    // Partition A: model has mastered it (utility drops to 0.05, seen 25 epochs)
    lake.record_training_feedback("market", "partition_a", 0.05, Some(0.0), 25)
        .unwrap();

    // Partition C: model still has high reconstruction error (utility remains 0.90, seen only 2 epochs)
    lake.record_training_feedback("market", "partition_c", 0.90, Some(0.2), 2)
        .unwrap();

    // Run utility-based pruning: threshold 0.15
    let evicted = lake
        .prune_low_utility(0.15)
        .expect("Prune low utility failed");
    assert_eq!(
        evicted, 1,
        "Only partition A should be pruned as model won't benefit from it"
    );

    // Verify partition A is gone, but crisis_2020 and partition_c are strictly preserved
    assert!(lake.load_parquet("market", "partition_a", 1010).is_err());
    assert!(lake.load_parquet("market", "crisis_2020", 1010).is_ok());
    assert!(lake.load_parquet("market", "partition_c", 1010).is_ok());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_parallel_batch_ingestion_and_verification() {
    let temp_dir =
        std::env::temp_dir().join(format!("spodeian_parallel_test_{}", std::process::id()));
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: 10 * 1024 * 1024,
        default_ttl_seconds: None,
    };

    let mut lake = WindowsDataLake::new(config).expect("Failed to initialize lake");
    let dataset1 = create_mock_market_dataset();
    let dataset2 = create_mock_market_dataset();
    let dataset3 = create_mock_market_dataset();

    let batch = vec![
        ("asx", "part_1", &dataset1, None, 0.1, false),
        ("asx", "part_2", &dataset2, None, 0.2, false),
        ("asx", "part_3", &dataset3, None, 0.9, true),
    ];

    // Parallel batch storage
    let metas = lake
        .store_parquet_batch_parallel(&batch, 1000)
        .expect("Parallel batch store failed");
    assert_eq!(metas.len(), 3);
    assert_eq!(lake.summary().total_partitions, 3);

    // Parallel partition integrity verification
    let to_verify = vec![("asx", "part_1"), ("asx", "part_2"), ("asx", "part_3")];
    let verified = lake
        .verify_partitions_parallel(&to_verify)
        .expect("Parallel verification failed");
    assert!(verified);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_debounced_wal_persistence_and_replay() {
    let temp_dir = std::env::temp_dir().join(format!("spodeian_wal_test_{}", std::process::id()));
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: 10 * 1024 * 1024,
        default_ttl_seconds: None,
    };

    let dataset = create_mock_market_dataset();

    {
        let mut lake = WindowsDataLake::new(config.clone()).expect("Failed to initialize lake");
        lake.store_parquet("market", "tick_partition", &dataset, None, 1000)
            .unwrap();

        // Perform fast training feedback updates that write to append-only WAL
        lake.record_training_feedback("market", "tick_partition", 0.42, Some(0.15), 5)
            .unwrap();
        lake.set_pinned("market", "tick_partition", true).unwrap();

        // Check that wal file exists before full flush
        let wal_path = temp_dir.join("lake_registry.wal");
        assert!(
            wal_path.exists(),
            "WAL file should be created for fast append"
        );
    }

    // Now re-open data lake from disk: should replay WAL entries seamlessly
    {
        let mut lake =
            WindowsDataLake::new(config).expect("Failed to re-initialize lake with WAL replay");
        let meta = lake
            .load_parquet("market", "tick_partition", 1010)
            .expect("Load after WAL replay failed");
        assert_eq!(meta.num_rows, 3);

        // Verify replayed in-memory state
        let retrieved = lake.summary();
        assert_eq!(retrieved.total_partitions, 1);

        // Explicitly flush / compact WAL
        lake.flush_wal().expect("Flush WAL failed");
        let wal_path = temp_dir.join("lake_registry.wal");
        assert!(
            !wal_path.exists(),
            "WAL file should be removed upon compaction"
        );
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_load_partitions_range_streaming() {
    let temp_dir = std::env::temp_dir().join(format!("spodeian_range_test_{}", std::process::id()));
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: 10 * 1024 * 1024,
        default_ttl_seconds: None,
    };

    let mut lake = WindowsDataLake::new(config).expect("Failed to initialize lake");
    let dataset1 = create_mock_market_dataset();
    let dataset2 = create_mock_market_dataset();

    // Store across different epochs
    lake.store_parquet("ticks", "day_1", &dataset1, None, 1000)
        .unwrap();
    lake.store_parquet("ticks", "day_2", &dataset2, None, 2000)
        .unwrap();
    lake.store_parquet("ticks", "day_3", &dataset1, None, 5000)
        .unwrap();

    // Query range [500, 2500]: should load day_1 and day_2 (total 6 rows)
    let range_data = lake
        .load_partitions_range("ticks", 500, 2500)
        .expect("Range query failed");
    assert_eq!(range_data.num_rows, 6);
    assert_eq!(range_data.float_columns.get("close").unwrap().len(), 6);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_citation_distillation_formats() {
    use spodeian_lake::registry::DatasetMetadata;
    use spodeian_lake::{CitationDistiller, CitationFormat};
    use std::path::PathBuf;

    let mut meta = DatasetMetadata {
        dataset_id: "asx_order_book".to_string(),
        partition_key: "2026_09_28".to_string(),
        file_path: PathBuf::from("data/asx.parquet"),
        record_count: 50000,
        byte_size: 1048576,
        sha256_checksum: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_string(),
        created_at_epoch_s: 1780000000,
        last_accessed_epoch_s: 1780000000,
        ttl_seconds: Some(86400),
        author: Some("Australian Financial Markets Research Lab".to_string()),
        doi_or_url: Some("https://doi.org/10.1000/182".to_string()),
        license: Some("CC-BY-4.0".to_string()),
        training_utility: 0.85,
        anomaly_score: 0.12,
        training_epochs_seen: 4,
        is_pinned: false,
    };

    // 1. BibTeX
    let bibtex = CitationDistiller::distill(&meta, CitationFormat::BibTeX);
    assert!(bibtex.contains("@misc{asx_order_book_2026_09_28,"));
    assert!(bibtex.contains("author = {Australian Financial Markets Research Lab}"));
    assert!(bibtex.contains("license = {CC-BY-4.0}"));
    assert!(bibtex.contains("SHA256: e3b0c442"));

    // 2. CFF
    let cff = CitationDistiller::distill(&meta, CitationFormat::Cff);
    assert!(cff.contains("title: \"Dataset: asx_order_book - Partition: 2026_09_28\""));
    assert!(cff.contains("license: \"CC-BY-4.0\""));
    assert!(cff.contains("sha256:e3b0c442"));

    // 3. APA
    let apa = CitationDistiller::distill(&meta, CitationFormat::Apa);
    assert!(apa.contains("Australian Financial Markets Research Lab"));
    assert!(apa.contains("asx_order_book: 2026_09_28 [Data set]"));
    assert!(apa.contains("https://doi.org/10.1000/182"));

    // 4. JSON-LD
    let json_ld = CitationDistiller::distill(&meta, CitationFormat::JsonLd);
    assert!(json_ld.contains("\"@type\": \"Dataset\""));
    assert!(json_ld.contains("application/vnd.apache.parquet"));

    // Manifest distillation
    meta.partition_key = "2026_09_29".to_string();
    let manifest_cff = CitationDistiller::distill_manifest(&[meta.clone()], CitationFormat::Cff);
    assert!(manifest_cff.contains("cff-version: 1.2.0"));
    assert!(manifest_cff.contains("references:"));
}

#[test]
fn test_experiment_registry_lifecycle_and_wal() {
    use spodeian_lake::ExperimentRecord;
    use std::collections::HashMap;

    let temp_dir = std::env::temp_dir().join(format!("spodeian_exp_test_{}", std::process::id()));
    let config = DataLakeConfig {
        lake_root: temp_dir.clone(),
        max_storage_bytes: 10 * 1024 * 1024,
        default_ttl_seconds: Some(3600),
    };

    let mut lake = WindowsDataLake::new(config.clone()).expect("Failed to initialize data lake");

    let mut metrics_a = HashMap::new();
    metrics_a.insert("train_loss".to_string(), 0.042);
    metrics_a.insert("val_loss".to_string(), 0.048);
    metrics_a.insert("best_loss".to_string(), 0.045);

    let mut hw_context = HashMap::new();
    hw_context.insert("device".to_string(), "cpu".to_string());
    hw_context.insert("torch".to_string(), "2.6.0".to_string());

    let exp_a = ExperimentRecord {
        run_id: "run-uuid-001".to_string(),
        experiment_name: "risky_business_ssl_v1".to_string(),
        config_yaml_snapshot: "name: risky_business_ssl_v1\nepochs: 50".to_string(),
        final_metrics: metrics_a,
        checkpoint_path: Some(temp_dir.join("ckpt_001.pt")),
        checkpoint_sha256: Some("sha256_mock_hash_001".to_string()),
        hardware_context: hw_context.clone(),
        timestamp_epoch_s: 1700000000,
        duration_seconds: 124.5,
        completion_status: "COMPLETED".to_string(),
        tags: vec!["ssl".to_string(), "grokking".to_string()],
    };

    let mut metrics_b = HashMap::new();
    metrics_b.insert("train_loss".to_string(), 0.038);
    metrics_b.insert("val_loss".to_string(), 0.041);
    metrics_b.insert("best_loss".to_string(), 0.039);

    let exp_b = ExperimentRecord {
        run_id: "run-uuid-002".to_string(),
        experiment_name: "risky_business_ssl_v1".to_string(),
        config_yaml_snapshot: "name: risky_business_ssl_v1\nepochs: 100".to_string(),
        final_metrics: metrics_b,
        checkpoint_path: Some(temp_dir.join("ckpt_002.pt")),
        checkpoint_sha256: Some("sha256_mock_hash_002".to_string()),
        hardware_context: hw_context,
        timestamp_epoch_s: 1700000500,
        duration_seconds: 248.0,
        completion_status: "COMPLETED".to_string(),
        tags: vec!["ssl".to_string(), "pruned".to_string()],
    };

    // Register experiments
    lake.register_experiment(exp_a.clone())
        .expect("Failed to register experiment A");
    lake.register_experiment(exp_b.clone())
        .expect("Failed to register experiment B");

    // Query experiments
    let all = lake.list_experiments();
    assert_eq!(all.len(), 2);
    // Verified sorted by timestamp descending
    assert_eq!(all[0].run_id, "run-uuid-002");
    assert_eq!(all[1].run_id, "run-uuid-001");

    let by_name = lake.list_experiments_by_name("risky_business_ssl_v1");
    assert_eq!(by_name.len(), 2);

    let fetched = lake
        .get_experiment("run-uuid-001")
        .expect("Missing experiment");
    assert_eq!(fetched.experiment_name, "risky_business_ssl_v1");
    assert_eq!(fetched.final_metrics.get("val_loss").copied(), Some(0.048));

    // Test WAL persistence and reload across lake re-instantiation
    let lake2 = WindowsDataLake::new(config).expect("Failed to reload data lake");
    let all_reloaded = lake2.list_experiments();
    assert_eq!(all_reloaded.len(), 2);
    assert_eq!(all_reloaded[0].run_id, "run-uuid-002");

    let _ = std::fs::remove_dir_all(&temp_dir);
}
