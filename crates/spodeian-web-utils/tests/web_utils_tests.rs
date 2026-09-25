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
