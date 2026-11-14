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

#[test]
fn test_pwa_inline_script_hashes_and_finalise_static_site() {
    use spodeian_web_utils::pwa::{
        CSP_PLACEHOLDER, finalise_static_site, inline_script_hashes, precache_url,
    };

    let html = r#"<!doctype html>
<html>
<head>
  <script src="/pkg/app.js"></script>
  <script>console.log('boot');</script>
  <script></script>
</head>
<body>ok</body>
</html>"#;
    let hashes = inline_script_hashes(html);
    assert_eq!(hashes.len(), 1);
    assert!(!hashes[0].is_empty());

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("about")).unwrap();
    let index_path = root.join("index.html");
    let about_path = root.join("about").join("index.html");
    std::fs::write(&index_path, html).unwrap();
    std::fs::write(&about_path, "<html><body>About</body></html>").unwrap();
    std::fs::write(root.join("pkg.js"), "console.log(1);").unwrap();
    std::fs::write(root.join("notes.txt"), "license").unwrap();
    std::fs::write(
        root.join("_headers"),
        format!("/*\n  Content-Security-Policy: script-src 'self' {CSP_PLACEHOLDER}\n"),
    )
    .unwrap();
    std::fs::write(
        root.join("sw.js"),
        "const BUILD_ID = '__BUILD_ID__';\nconst PRECACHE_URLS = /*__PRECACHE_URLS__*/ [];\n",
    )
    .unwrap();

    assert_eq!(precache_url(root, &index_path), Some("/".to_string()));
    assert_eq!(precache_url(root, &about_path), Some("/about".to_string()));
    assert_eq!(precache_url(root, &root.join("notes.txt")), None);
    assert_eq!(precache_url(root, &root.join("sw.js")), None);

    let build_id = finalise_static_site(root, &[index_path, about_path])
        .unwrap()
        .expect("sw.js should be populated");
    assert_eq!(build_id.len(), 12);

    let headers_out = std::fs::read_to_string(root.join("_headers")).unwrap();
    assert!(!headers_out.contains(CSP_PLACEHOLDER));
    assert!(headers_out.contains(&format!("'sha256-{}'", hashes[0])));

    let sw_out = std::fs::read_to_string(root.join("sw.js")).unwrap();
    assert!(sw_out.contains(&format!("'{build_id}'")));
    assert!(sw_out.contains("'/'"));
    assert!(sw_out.contains("'/about'"));
    assert!(sw_out.contains("'/pkg.js'"));
    assert!(!sw_out.contains("notes.txt"));
}
