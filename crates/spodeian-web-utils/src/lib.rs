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

pub mod pwa;

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

/// Detected host or browser runtime environment across Desktop, Tauri, and WebAssembly Edge targets.
#[doc = "**Stability: Stable** — cross-target runtime detection"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTarget {
    /// Native desktop, CLI, or static SSR generator host.
    NativeHost,
    /// Desktop/mobile application embedded inside a Tauri v2 webview (`window.__TAURI_INTERNALS__`).
    TauriWebview,
    /// Standard serverless WebAssembly browser tab (`window` without Tauri internals).
    WebBrowser,
    /// WebAssembly environment without a DOM `window` (e.g. Web Worker / edge isolate).
    HeadlessWorker,
}

impl RuntimeTarget {
    /// Detects the active execution environment at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            if is_tauri_desktop() {
                Self::TauriWebview
            } else if web_sys::window().is_some() {
                Self::WebBrowser
            } else {
                Self::HeadlessWorker
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::NativeHost
        }
    }

    /// Human-readable display label for diagnostics UI and telemetry logs.
    pub const fn label(self) -> &'static str {
        match self {
            Self::NativeHost => "Native Host",
            Self::TauriWebview => "Tauri v2 Webview",
            Self::WebBrowser => "WebAssembly Browser",
            Self::HeadlessWorker => "WASM Headless Worker",
        }
    }
}

/// Returns `true` when running inside a Tauri v2 desktop/mobile webview (`window.__TAURI_INTERNALS__`
/// or legacy `window.__TAURI__`), and `false` on standard browsers or native hosts.
#[doc = "**Stability: Stable** — Tauri v2 runtime detection"]
pub fn is_tauri_desktop() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let Some(window) = web_sys::window() else {
            return false;
        };
        let has_internals = js_sys::Reflect::get(
            &window,
            &wasm_bindgen::JsValue::from_str("__TAURI_INTERNALS__"),
        )
        .map(|val| !val.is_undefined() && !val.is_null())
        .unwrap_or(false);
        if has_internals {
            return true;
        }
        js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__TAURI__"))
            .map(|val| !val.is_undefined() && !val.is_null())
            .unwrap_or(false)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// Returns `true` when executing inside a standard browser tab (excluding Tauri webviews and native hosts).
#[doc = "**Stability: Stable** — browser target detection"]
pub fn is_wasm_browser() -> bool {
    matches!(RuntimeTarget::detect(), RuntimeTarget::WebBrowser)
}

/// Returns `true` when WebGPU (`navigator.gpu` on `wasm32`, or native Vulkan/Metal/DX12 backends)
/// is potentially available on the current runtime target.
#[doc = "**Stability: Stable** — WebGPU capability probe"]
pub fn has_webgpu_support() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let Some(window) = web_sys::window() else {
            return false;
        };
        let navigator = window.navigator();
        js_sys::Reflect::get(&navigator, &wasm_bindgen::JsValue::from_str("gpu"))
            .map(|val| !val.is_undefined() && !val.is_null())
            .unwrap_or(false)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

/// Returns logical CPU concurrency (`navigator.hardwareConcurrency` on `wasm32`, or
/// `std::thread::available_parallelism()` on native targets), clamped to at least `1`.
#[doc = "**Stability: Stable** — cross-target concurrency probe"]
pub fn hardware_concurrency() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| {
                js_sys::Reflect::get(
                    &w.navigator(),
                    &wasm_bindgen::JsValue::from_str("hardwareConcurrency"),
                )
                .ok()
            })
            .and_then(|v| v.as_f64())
            .filter(|n| n.is_finite() && *n >= 1.0)
            .map(|n| n.round() as usize)
            .unwrap_or(1)
            .max(1)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    }
}

/// Invokes a backend Tauri v2 command (`command`) with JSON arguments (`args_json`) from a
/// `wasm32` webview, returning the JSON-serialized response payload.
///
/// Native host builds return [`WebUtilsError::Unavailable`] since IPC is only needed across the
/// webview boundary.
#[doc = "**Stability: Stable** — Tauri v2 webview IPC helper"]
pub async fn invoke_tauri_ipc(command: &str, args_json: &str) -> Result<String, WebUtilsError> {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;

        let window =
            web_sys::window().ok_or_else(|| WebUtilsError::Unavailable("no window".into()))?;
        let internals = js_sys::Reflect::get(
            &window,
            &wasm_bindgen::JsValue::from_str("__TAURI_INTERNALS__"),
        )
        .ok()
        .filter(|v| !v.is_undefined() && !v.is_null())
        .ok_or_else(|| WebUtilsError::Unavailable("not running in a Tauri webview".into()))?;

        let invoke_fn =
            js_sys::Reflect::get(&internals, &wasm_bindgen::JsValue::from_str("invoke"))
                .map_err(|e| js_failure("reading Tauri invoke", e))?
                .dyn_into::<js_sys::Function>()
                .map_err(|e| js_failure("resolving Tauri invoke function", e))?;

        let payload_src = if args_json.trim().is_empty() {
            "{}"
        } else {
            args_json
        };
        let parsed_args =
            js_sys::JSON::parse(payload_src).map_err(|e| js_failure("parsing IPC args JSON", e))?;
        let promise_val = invoke_fn
            .call2(
                &internals,
                &wasm_bindgen::JsValue::from_str(command),
                &parsed_args,
            )
            .map_err(|e| js_failure("invoking Tauri command", e))?;

        let promise: js_sys::Promise = promise_val
            .dyn_into()
            .map_err(|e| js_failure("converting Tauri IPC return to Promise", e))?;
        let result = wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| js_failure("awaiting Tauri IPC response", e))?;

        if result.is_undefined() || result.is_null() {
            return Ok("null".to_string());
        }
        let json_str = js_sys::JSON::stringify(&result)
            .map_err(|e| js_failure("stringifying Tauri IPC response", e))?;
        Ok(json_str.as_string().unwrap_or_else(|| "null".to_string()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (command, args_json);
        Err(WebUtilsError::Unavailable(
            "Tauri webview IPC requires a wasm32 webview runtime".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_detection_and_capabilities_work_on_host() {
        assert_eq!(RuntimeTarget::detect(), RuntimeTarget::NativeHost);
        assert_eq!(RuntimeTarget::NativeHost.label(), "Native Host");
        assert!(!is_tauri_desktop());
        assert!(!is_wasm_browser());
        assert!(has_webgpu_support());
        assert!(hardware_concurrency() >= 1);
    }
}
