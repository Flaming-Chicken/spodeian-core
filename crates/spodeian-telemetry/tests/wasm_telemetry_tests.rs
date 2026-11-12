//! Browser tests: the ring buffer and metrics must capture events on WebAssembly too.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn browser_events_reach_ring_buffer_and_metrics() {
    let buffer = spodeian_telemetry::init_telemetry_with_buffer("wasm_test", "info");
    let before = spodeian_telemetry::metrics_snapshot();

    tracing::warn!(probe = "wasm", "browser telemetry probe");

    let after = spodeian_telemetry::metrics_snapshot();
    assert!(after.warnings > before.warnings);
    assert!(
        buffer
            .snapshot()
            .iter()
            .any(|e| e.message == "browser telemetry probe" && e.level == "WARN")
    );
    assert!(after.uptime_millis < 60_000);
}
