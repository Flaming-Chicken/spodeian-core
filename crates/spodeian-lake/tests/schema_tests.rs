use spodeian_lake::{
    AcousticTelemetrySnapshot, FinancialObservation, PlayerProgressEvent, TabularDataset,
};

#[test]
fn test_financial_observation_roundtrip() {
    let records = vec![
        FinancialObservation {
            timestamp: 1700000000,
            symbol: "BHP.AX".to_string(),
            open: 42.10,
            high: 42.80,
            low: 42.05,
            close: 42.60,
            volume: 150000.0,
            vwap: 42.45,
            spread: 0.02,
        },
        FinancialObservation {
            timestamp: 1700000060,
            symbol: "BHP.AX".to_string(),
            open: 42.60,
            high: 42.95,
            low: 42.55,
            close: 42.90,
            volume: 210000.0,
            vwap: 42.75,
            spread: 0.03,
        },
    ];

    let dataset = FinancialObservation::to_tabular_dataset(&records).expect("Failed to convert to tabular dataset");
    assert_eq!(dataset.num_rows, 2);

    // Parquet buffer roundtrip
    let mut buffer = Vec::new();
    dataset.write_to_parquet(&mut buffer).expect("Write to parquet failed");
    let restored_dataset = TabularDataset::read_from_parquet(&buffer).expect("Read from parquet failed");

    let restored_records = FinancialObservation::from_tabular_dataset(&restored_dataset)
        .expect("Failed to parse financial observations");

    assert_eq!(restored_records.len(), 2);
    assert_eq!(restored_records, records);
}

#[test]
fn test_acoustic_telemetry_snapshot_roundtrip() {
    let snapshots = vec![
        AcousticTelemetrySnapshot {
            timestamp: 1700000100,
            frame_idx: 0,
            rms_db: -18.5,
            peak_db: -6.2,
            spectral_centroid: 1250.0,
            zero_crossing_rate: 0.045,
            active_voices: 8,
            buffer_underruns: 0,
        },
        AcousticTelemetrySnapshot {
            timestamp: 1700000101,
            frame_idx: 1,
            rms_db: -16.2,
            peak_db: -4.1,
            spectral_centroid: 1480.0,
            zero_crossing_rate: 0.052,
            active_voices: 12,
            buffer_underruns: 1,
        },
    ];

    let dataset = AcousticTelemetrySnapshot::to_tabular_dataset(&snapshots)
        .expect("Failed to convert acoustic snapshots to dataset");
    assert_eq!(dataset.num_rows, 2);

    let mut buffer = Vec::new();
    dataset.write_to_parquet(&mut buffer).expect("Write to parquet failed");
    let restored_dataset = TabularDataset::read_from_parquet(&buffer).expect("Read from parquet failed");

    let restored_snapshots = AcousticTelemetrySnapshot::from_tabular_dataset(&restored_dataset)
        .expect("Failed to parse acoustic snapshots");

    assert_eq!(restored_snapshots.len(), 2);
    assert_eq!(restored_snapshots, snapshots);
}

#[test]
fn test_player_progress_event_roundtrip() {
    let events = vec![
        PlayerProgressEvent {
            timestamp: 1700000200,
            session_id: "sess_42".to_string(),
            realm_level: 3,
            void_resonance: 0.72,
            soul_shards: 150,
            mana_ratio: 0.88,
            event_type: "void_tasting".to_string(),
        },
        PlayerProgressEvent {
            timestamp: 1700000250,
            session_id: "sess_42".to_string(),
            realm_level: 4,
            void_resonance: 0.89,
            soul_shards: 220,
            mana_ratio: 0.95,
            event_type: "breakthrough".to_string(),
        },
    ];

    let dataset = PlayerProgressEvent::to_tabular_dataset(&events)
        .expect("Failed to convert player events to dataset");
    assert_eq!(dataset.num_rows, 2);

    let mut buffer = Vec::new();
    dataset.write_to_parquet(&mut buffer).expect("Write to parquet failed");
    let restored_dataset = TabularDataset::read_from_parquet(&buffer).expect("Read from parquet failed");

    let restored_events = PlayerProgressEvent::from_tabular_dataset(&restored_dataset)
        .expect("Failed to parse player events");

    assert_eq!(restored_events.len(), 2);
    assert_eq!(restored_events, events);
}

#[test]
fn test_empty_dataset_handling() {
    let empty_financial: Vec<FinancialObservation> = Vec::new();
    let dataset = FinancialObservation::to_tabular_dataset(&empty_financial).unwrap();
    assert_eq!(dataset.num_rows, 0);
    let parsed = FinancialObservation::from_tabular_dataset(&dataset).unwrap();
    assert!(parsed.is_empty());

    let empty_acoustic: Vec<AcousticTelemetrySnapshot> = Vec::new();
    let dataset = AcousticTelemetrySnapshot::to_tabular_dataset(&empty_acoustic).unwrap();
    assert_eq!(dataset.num_rows, 0);
    let parsed = AcousticTelemetrySnapshot::from_tabular_dataset(&dataset).unwrap();
    assert!(parsed.is_empty());

    let empty_player: Vec<PlayerProgressEvent> = Vec::new();
    let dataset = PlayerProgressEvent::to_tabular_dataset(&empty_player).unwrap();
    assert_eq!(dataset.num_rows, 0);
    let parsed = PlayerProgressEvent::from_tabular_dataset(&dataset).unwrap();
    assert!(parsed.is_empty());
}
