//! Universal web DOM interaction utilities for Spodeian applications.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides transparent fallback between browser DOM actions on `wasm32`
//! (Blob URL triggers, navigator.clipboard) and native file/clipboard APIs on desktop.
//!
//! Since v1.1, `try_*` functions and [`write_clipboard_text`] report failures, so an
//! interface can tell users the truth instead of assuming success. The original
//! fire-and-forget functions remain and log their errors.

#[allow(unused_imports)]
use tracing::{error, info};

/// Why a download or clipboard operation failed.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebUtilsError {
    /// The browser API needed is not available here (no window, no clipboard, or a
    /// native build without a clipboard implementation).
    Unavailable(String),
    /// The browser or operating system refused or failed the operation.
    Failed(String),
}

impl std::fmt::Display for WebUtilsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(msg) => write!(f, "unavailable: {msg}"),
            Self::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

impl std::error::Error for WebUtilsError {}

#[cfg(target_arch = "wasm32")]
fn js_failure(context: &str, value: wasm_bindgen::JsValue) -> WebUtilsError {
    use wasm_bindgen::JsCast;
    let detail = value
        .dyn_ref::<js_sys::Error>()
        .map(|e| String::from(e.message()))
        .or_else(|| value.as_string())
        .unwrap_or_else(|| format!("{value:?}"));
    WebUtilsError::Failed(format!("{context}: {detail}"))
}

/// Starts a browser download of `blob` named `filename`. The object URL is revoked a
/// second later: revoking it immediately can cancel the download in some browsers.
#[cfg(target_arch = "wasm32")]
fn download_blob(filename: &str, blob: &web_sys::Blob) -> Result<(), WebUtilsError> {
    use wasm_bindgen::JsCast;
    let window = web_sys::window().ok_or_else(|| WebUtilsError::Unavailable("no window".into()))?;
    let document = window
        .document()
        .ok_or_else(|| WebUtilsError::Unavailable("no document".into()))?;
    let url = web_sys::Url::create_object_url_with_blob(blob)
        .map_err(|e| js_failure("creating object URL", e))?;
    let anchor = document
        .create_element("a")
        .map_err(|e| js_failure("creating link", e))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|e| js_failure("creating link", e.into()))?;
    anchor.set_href(&url);
    anchor.set_download(filename);
    anchor.click();

    let revoke = wasm_bindgen::closure::Closure::once_into_js(move || {
        let _ = web_sys::Url::revoke_object_url(&url);
    });
    window
        .set_timeout_with_callback_and_timeout_and_arguments_0(revoke.unchecked_ref(), 1000)
        .map_err(|e| js_failure("scheduling cleanup", e))?;
    Ok(())
}

/// Triggers a text file download (JSON, CSV, markdown, etc.), reporting failure.
///
/// On native targets the file is written to `filename` relative to the working directory.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub fn try_trigger_text_download(
    filename: &str,
    content: &str,
    mime_type: &str,
) -> Result<(), WebUtilsError> {
    #[cfg(target_arch = "wasm32")]
    {
        let blob_parts = js_sys::Array::new();
        blob_parts.push(&wasm_bindgen::JsValue::from_str(content));
        let blob_props = web_sys::BlobPropertyBag::new();
        blob_props.set_type(mime_type);
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_props)
            .map_err(|e| js_failure("creating file", e))?;
        download_blob(filename, &blob)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime_type;
        std::fs::write(filename, content)
            .map_err(|e| WebUtilsError::Failed(format!("writing '{filename}': {e}")))
    }
}

/// Triggers a binary file download (e.g. compressed BSON, Parquet), reporting failure.
///
/// On native targets the file is written to `filename` relative to the working directory.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub fn try_trigger_binary_download(
    filename: &str,
    bytes: &[u8],
    mime_type: &str,
) -> Result<(), WebUtilsError> {
    #[cfg(target_arch = "wasm32")]
    {
        let uint8_array = js_sys::Uint8Array::from(bytes);
        let blob_parts = js_sys::Array::new();
        blob_parts.push(&uint8_array.buffer());
        let blob_props = web_sys::BlobPropertyBag::new();
        blob_props.set_type(mime_type);
        let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&blob_parts, &blob_props)
            .map_err(|e| js_failure("creating file", e))?;
        download_blob(filename, &blob)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime_type;
        std::fs::write(filename, bytes)
            .map_err(|e| WebUtilsError::Failed(format!("writing '{filename}': {e}")))
    }
}

/// Writes `text` to the system clipboard and resolves only once the browser confirms it.
///
/// Browsers refuse clipboard writes without a recent user gesture or permission, so callers
/// should show success only when this returns `Ok`. Native builds have no clipboard
/// implementation and return [`WebUtilsError::Unavailable`].
#[doc = "**Stability: Stable** — semver-guaranteed since v1.1"]
pub async fn write_clipboard_text(text: &str) -> Result<(), WebUtilsError> {
    #[cfg(target_arch = "wasm32")]
    {
        let window =
            web_sys::window().ok_or_else(|| WebUtilsError::Unavailable("no window".into()))?;
        let promise = window.navigator().clipboard().write_text(text);
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| js_failure("writing to clipboard", e))?;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = text;
        Err(WebUtilsError::Unavailable(
            "no native clipboard implementation".into(),
        ))
    }
}

/// Triggers a client-side text file download (JSON, CSV, markdown, etc.)
///
/// Failures are logged, not returned; prefer [`try_trigger_text_download`].
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn trigger_text_download(filename: &str, content: &str, mime_type: &str) {
    match try_trigger_text_download(filename, content, mime_type) {
        Ok(()) => info!("Started download of '{}'", filename),
        Err(e) => error!("Download of '{}' failed: {}", filename, e),
    }
}

/// Triggers a client-side binary file download (e.g. Compressed BSON, Parquet, Safetensors)
///
/// Failures are logged, not returned; prefer [`try_trigger_binary_download`].
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn trigger_binary_download(filename: &str, bytes: &[u8], mime_type: &str) {
    match try_trigger_binary_download(filename, bytes, mime_type) {
        Ok(()) => info!("Started download of '{}'", filename),
        Err(e) => error!("Download of '{}' failed: {}", filename, e),
    }
}

/// Copies text into the system clipboard.
///
/// Fire-and-forget: the browser may still refuse the write, and failures are only logged.
/// Prefer [`write_clipboard_text`], which reports the outcome.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn copy_to_clipboard(text: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        let text = text.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            match write_clipboard_text(&text).await {
                Ok(()) => info!("Copied {} characters to browser clipboard", text.len()),
                Err(e) => error!("Clipboard write failed: {}", e),
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        info!("Native clipboard requested ({} bytes)", text.len());
    }
}
