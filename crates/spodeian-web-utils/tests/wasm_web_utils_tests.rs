//! Browser tests: downloads and clipboard writes report their real outcome.
#![cfg(target_arch = "wasm32")]

use spodeian_web_utils::{
    WebUtilsError, try_trigger_binary_download, try_trigger_text_download, write_clipboard_text,
};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn downloads_start_in_the_browser() {
    assert_eq!(
        try_trigger_text_download("a.json", "{}", "application/json"),
        Ok(())
    );
    assert_eq!(
        try_trigger_binary_download("b.bin", &[1, 2, 3], "application/octet-stream"),
        Ok(())
    );
}

#[wasm_bindgen_test]
async fn clipboard_write_reports_the_browser_outcome() {
    // A headless test page has no user gesture, so the browser may refuse the write. What
    // matters is that a refusal is reported as `Failed` rather than assumed to succeed.
    match write_clipboard_text("probe").await {
        Ok(()) | Err(WebUtilsError::Failed(_)) => {}
        Err(other) => panic!("unexpected clipboard result: {other:?}"),
    }
}
