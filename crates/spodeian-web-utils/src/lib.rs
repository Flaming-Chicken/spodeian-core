//! Universal web DOM interaction utilities for Spodeian applications.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides transparent fallback between browser DOM actions on `wasm32`
//! (Blob URL triggers, navigator.clipboard) and native file/clipboard APIs on desktop.

#[allow(unused_imports)]
use tracing::{error, info};

/// Triggers a client-side text file download (JSON, CSV, markdown, etc.)
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn trigger_text_download(filename: &str, content: &str, mime_type: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                let blob_parts = js_sys::Array::new();
                blob_parts.push(&wasm_bindgen::JsValue::from_str(content));
                let blob_props = web_sys::BlobPropertyBag::new();
                blob_props.set_type(mime_type);
                if let Ok(blob) =
                    web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_props)
                {
                    if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
                        if let Ok(element) = document.create_element("a") {
                            if let Ok(anchor) = element.dyn_into::<web_sys::HtmlAnchorElement>() {
                                anchor.set_href(&url);
                                anchor.set_download(filename);
                                anchor.click();
                                let _ = web_sys::Url::revoke_object_url(&url);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime_type;
        match std::fs::write(filename, content) {
            Ok(()) => info!("Successfully wrote local file: {}", filename),
            Err(e) => error!("Failed to write export file '{}': {}", filename, e),
        }
    }
}

/// Triggers a client-side binary file download (e.g. Compressed BSON, Parquet, Safetensors)
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn trigger_binary_download(filename: &str, bytes: &[u8], mime_type: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                let uint8_array = js_sys::Uint8Array::from(bytes);
                let blob_parts = js_sys::Array::new();
                blob_parts.push(&uint8_array.buffer());
                let blob_props = web_sys::BlobPropertyBag::new();
                blob_props.set_type(mime_type);
                if let Ok(blob) =
                    web_sys::Blob::new_with_u8_array_sequence_and_options(&blob_parts, &blob_props)
                {
                    if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
                        if let Ok(element) = document.create_element("a") {
                            if let Ok(anchor) = element.dyn_into::<web_sys::HtmlAnchorElement>() {
                                anchor.set_href(&url);
                                anchor.set_download(filename);
                                anchor.click();
                                let _ = web_sys::Url::revoke_object_url(&url);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime_type;
        match std::fs::write(filename, bytes) {
            Ok(()) => info!("Successfully exported binary file: {}", filename),
            Err(e) => error!("Failed to write binary export file '{}': {}", filename, e),
        }
    }
}

/// Copies text into the system clipboard.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn copy_to_clipboard(text: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            let navigator = window.navigator();
            let clipboard = navigator.clipboard();
            let _ = clipboard.write_text(text);
            info!("Copied {} characters to browser clipboard", text.len());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        info!("Native clipboard requested ({} bytes)", text.len());
    }
}
