#![cfg(not(target_arch = "wasm32"))]

use spodeian_web_utils::{copy_to_clipboard, trigger_binary_download, trigger_text_download};

#[test]
fn test_web_utils_native_fallback() {
    let tmp_file = std::env::temp_dir().join("test_download_export.txt");
    let tmp_path = tmp_file.to_string_lossy().to_string();

    trigger_text_download(&tmp_path, "sample text content", "text/plain");
    assert!(tmp_file.exists());
    let read_back = std::fs::read_to_string(&tmp_file).unwrap();
    assert_eq!(read_back, "sample text content");
    let _ = std::fs::remove_file(&tmp_file);

    let bin_file = std::env::temp_dir().join("test_binary_export.bin");
    let bin_path = bin_file.to_string_lossy().to_string();
    trigger_binary_download(&bin_path, &[1, 2, 3, 4, 5], "application/octet-stream");
    assert!(bin_file.exists());
    let read_bin = std::fs::read(&bin_file).unwrap();
    assert_eq!(read_bin, &[1, 2, 3, 4, 5]);
    let _ = std::fs::remove_file(&bin_file);

    copy_to_clipboard("test clipboard text");
}

#[test]
fn test_try_downloads_report_success_and_failure() {
    use spodeian_web_utils::{
        WebUtilsError, try_trigger_binary_download, try_trigger_text_download,
    };

    let file = std::env::temp_dir().join(format!("spodeian_try_{}.txt", std::process::id()));
    let path = file.to_string_lossy().to_string();
    assert_eq!(try_trigger_text_download(&path, "ok", "text/plain"), Ok(()));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "ok");
    let _ = std::fs::remove_file(&file);

    let missing_dir = std::env::temp_dir()
        .join("spodeian_no_such_dir")
        .join("x.bin")
        .to_string_lossy()
        .to_string();
    assert!(matches!(
        try_trigger_binary_download(&missing_dir, &[1], "application/octet-stream"),
        Err(WebUtilsError::Failed(_))
    ));
}

#[test]
fn test_native_clipboard_reports_unavailable() {
    use spodeian_web_utils::{WebUtilsError, write_clipboard_text};

    let result = pollster::block_on(write_clipboard_text("text"));
    assert!(matches!(result, Err(WebUtilsError::Unavailable(_))));
}
