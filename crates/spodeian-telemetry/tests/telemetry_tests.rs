use spodeian_telemetry::{
    LogEntry, LogRingBuffer, init_default, init_telemetry, init_telemetry_with_buffer,
};

#[test]
fn test_telemetry_initialization() {
    // Calling multiple times should not panic due to try_init internally
    init_default();
    init_telemetry("debug");
    tracing::info!("Test log from spodeian-telemetry integration test");
}

#[test]
fn test_log_ring_buffer_operations() {
    let buffer = LogRingBuffer::new(3);
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());

    let entry1 = LogEntry {
        timestamp_millis: 1000,
        level: "INFO".to_string(),
        target: "app::test".to_string(),
        message: "First entry".to_string(),
        fields: vec![("stage".to_string(), "init".to_string())],
    };

    let entry2 = LogEntry {
        timestamp_millis: 2000,
        level: "WARN".to_string(),
        target: "app::test".to_string(),
        message: "Second entry".to_string(),
        fields: vec![],
    };

    let entry3 = LogEntry {
        timestamp_millis: 3000,
        level: "ERROR".to_string(),
        target: "app::test".to_string(),
        message: "Third entry".to_string(),
        fields: vec![],
    };

    let entry4 = LogEntry {
        timestamp_millis: 4000,
        level: "INFO".to_string(),
        target: "app::test".to_string(),
        message: "Fourth entry (evicts first)".to_string(),
        fields: vec![],
    };

    buffer.push(entry1);
    buffer.push(entry2);
    buffer.push(entry3);
    assert_eq!(buffer.len(), 3);

    // Push 4th: should evict entry1
    buffer.push(entry4);
    assert_eq!(buffer.len(), 3);

    let snapshot = buffer.snapshot();
    assert_eq!(snapshot.len(), 3);
    assert_eq!(snapshot[0].message, "Second entry");
    assert_eq!(snapshot[1].message, "Third entry");
    assert_eq!(snapshot[2].message, "Fourth entry (evicts first)");

    buffer.clear();
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn test_init_telemetry_with_buffer() {
    let buffer = init_telemetry_with_buffer("test_app", "info");
    tracing::info!(
        test_field = "provenance",
        "Integration event testing ring buffer"
    );

    // Give subscriber a moment to capture
    let snapshot = buffer.snapshot();
    assert!(!snapshot.is_empty());
}

#[test]
fn test_metrics_snapshot_counts_events_by_level() {
    let _ = init_telemetry_with_buffer("metrics_test", "trace");
    let before = spodeian_telemetry::metrics_snapshot();

    tracing::warn!("metrics test warning");
    tracing::error!("metrics test error");

    let after = spodeian_telemetry::metrics_snapshot();
    assert!(after.warnings > before.warnings);
    assert!(after.errors > before.errors);
    assert_eq!(
        after.total_events(),
        after.errors + after.warnings + after.infos + after.debugs + after.traces
    );
    assert!(after.buffered_entries >= 1);
    assert!(after.buffered_entries <= after.buffer_capacity);
    assert_eq!(after.buffer_capacity, 2000);
}

#[test]
fn test_ring_buffer_reports_capacity() {
    assert_eq!(LogRingBuffer::new(5).capacity(), 5);
    // Zero is clamped to one so the buffer can always hold the latest entry.
    assert_eq!(LogRingBuffer::new(0).capacity(), 1);
}
