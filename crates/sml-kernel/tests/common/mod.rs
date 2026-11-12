use std::path::{Path, PathBuf};

/// Locates a weights file relative to either repository root or crate directory.
pub fn get_weights_path(rel: &str) -> PathBuf {
    let p1 = Path::new("weights").join(rel);
    if p1.exists() {
        return p1;
    }
    let p2 = Path::new("../../weights").join(rel);
    if p2.exists() {
        return p2;
    }
    panic!("Could not locate weights file: {}", rel);
}
