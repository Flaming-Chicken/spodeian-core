use spodeian_telemetry::{init_default, init_telemetry};

#[test]
fn test_telemetry_initialization() {
    // Calling multiple times should not panic due to try_init internally
    init_default();
    init_telemetry("debug");
    tracing::info!("Test log from spodeian-telemetry integration test");
}
