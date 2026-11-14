//! Lazy directory traversal and text-only tree serialization.
//!
//! Salvaged from `dir2txt` to provide deterministic, lazy UTF-8 directory crawling
//! that skips binary files and prunes empty directories.

use serde::{Serialize, Serializer};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Reads bytes from `reader` into `buf` until `buf` is full or EOF is reached.
fn read_to_fill_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<usize, std::io::Error> {
    let mut total = 0;
    while total < buf.len() {
        match reader.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(total)
}

/// Heuristic check for UTF-8 text: rejects NUL bytes and invalid UTF-8 sequences
/// (tolerating a split multi-byte sequence only at an exact buffer boundary).
fn is_utf8_text(sample: &[u8], hit_buffer_limit: bool) -> bool {
    if sample.contains(&0) {
        return false;
    }
    match std::str::from_utf8(sample) {
        Ok(_) => true,
        Err(e) => hit_buffer_limit && e.error_len().is_none(),
    }
}

/// A lazily-loaded file representation that defers reading full contents until requested.
#[derive(Clone, Debug, Default)]
pub struct LazyFile {
    /// The base filename within its parent directory.
    pub name: String,
    is_text: OnceLock<bool>,
    content: OnceLock<Option<String>>,
}

impl LazyFile {
    /// Creates a new [`LazyFile`] with deferred text detection and content loading.
    pub fn new(name: String) -> Self {
        Self {
            name,
            is_text: OnceLock::new(),
            content: OnceLock::new(),
        }
    }

    /// Computes the full path of this file relative to `parent_path`.
    pub fn path<P: AsRef<Path>>(&self, parent_path: P) -> PathBuf {
        parent_path.as_ref().join(&self.name)
    }

    /// Reads up to 1024 bytes from `reader` to determine whether the stream is UTF-8 text.
    ///
    /// Returns the number of bytes read and the filled prefix buffer so callers can
    /// avoid re-reading the header.
    pub fn set_is_text<R: Read>(
        &self,
        reader: &mut R,
    ) -> Result<(usize, [u8; 1024]), std::io::Error> {
        let mut buf = [0u8; 1024];
        let bytes_read = read_to_fill_or_eof(reader, buf.as_mut_slice())?;
        let is_text = is_utf8_text(&buf[..bytes_read], bytes_read == buf.len());

        if *self.is_text.get_or_init(|| is_text) == is_text {
            Ok((bytes_read, buf))
        } else {
            Err(std::io::Error::other("is_text state mismatch"))
        }
    }

    /// Returns the cached text status, or `None` if detection has not run yet.
    pub fn get_is_text(&self) -> Option<bool> {
        self.is_text.get().copied()
    }

    /// Loads the file content into memory if it is valid UTF-8 text.
    ///
    /// Idempotent when content is already cached; binary files are skipped cleanly.
    pub fn load_content<P: AsRef<Path>>(&self, parent_path: P) -> Result<(), std::io::Error> {
        if self.content.get().is_some() {
            return Ok(());
        }

        if let Some(false) = self.get_is_text() {
            if self.content.get().is_none() {
                let _ = self.set_content(None);
            }
            return Ok(());
        }

        let full_path = self.path(parent_path);
        let mut file = fs::File::open(full_path)?;
        let (bytes_read, buf) = self.set_is_text(&mut file)?;

        match self.get_is_text() {
            Some(true) => {
                let mut reader = std::io::BufReader::new(&mut file);
                let mut full_buf: Vec<u8> = buf[..bytes_read].to_vec();
                reader.read_to_end(&mut full_buf)?;

                self.set_content(Some(String::from_utf8_lossy(&full_buf).into_owned()))
                    .map_err(std::io::Error::other)?;
                Ok(())
            }
            Some(false) => {
                if self.content.get().is_none() {
                    let _ = self.set_content(None);
                }
                Ok(())
            }
            None => Err(std::io::Error::other("Failed to update is_text")),
        }
    }

    /// Manually initializes the file content and locks `is_text`.
    pub fn set_content(&self, content: Option<String>) -> Result<(), &'static str> {
        let is_some = content.is_some();
        if *self.is_text.get_or_init(|| is_some) != is_some {
            return Err("is_text already set differently");
        }
        self.content
            .set(content)
            .map_err(|_| "Content already set")?;
        Ok(())
    }

    /// Returns the cached UTF-8 content if loaded.
    pub fn get_content(&self) -> Option<&str> {
        self.content.get().and_then(|opt| opt.as_deref())
    }

    /// Returns the cached content if initialized, or computes it via `content_fn`.
    pub fn get_or_init_content(&self, content_fn: impl FnOnce() -> Option<String>) -> Option<&str> {
        self.content.get_or_init(content_fn).as_deref()
    }

    /// Returns `true` if UTF-8 content has been loaded into memory.
    pub fn is_text_ready(&self) -> bool {
        self.get_content().is_some()
    }

    /// Returns whether this file should be retained during pruning (`default` if undetected).
    pub fn prune(&self, default: bool) -> bool {
        *self.is_text.get().unwrap_or(&default)
    }
}

impl Serialize for LazyFile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(content) = self.get_content() {
            serializer.serialize_str(content)
        } else if let Some(&false) = self.is_text.get() {
            Err(serde::ser::Error::custom("File is not text."))
        } else {
            Err(serde::ser::Error::custom("Not initialized."))
        }
    }
}

/// A recursive directory tree containing lazily-loaded text files and subdirectories.
#[derive(Clone, Debug, Default)]
pub struct Directory {
    /// Directory name.
    pub name: String,
    /// Files directly inside this directory.
    pub files: Vec<LazyFile>,
    /// Nested subdirectories.
    pub directories: Vec<Directory>,
}

impl Directory {
    /// Creates an empty [`Directory`] with the given `name`.
    pub fn new(name: String) -> Self {
        Self {
            name,
            files: Vec::new(),
            directories: Vec::new(),
        }
    }

    /// Computes the path of this directory relative to `parent_path`.
    pub fn path<P: AsRef<Path>>(&self, parent_path: P) -> PathBuf {
        parent_path.as_ref().join(&self.name)
    }

    /// Walks `path`, skipping detected binary files and collapsing empty directories.
    ///
    /// If `lazy` is `true`, only the 1 KB text/binary header check is performed initially.
    pub fn from_path_slimmed<P: AsRef<Path>>(path: P, lazy: bool) -> std::io::Result<Self> {
        let path = path.as_ref();
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "root".to_string());

        let mut current_dir = Directory::new(name);

        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let Ok(entry) = entry else { continue };
                let entry_type = entry.file_type()?;

                if entry_type.is_dir() {
                    let dir = Self::from_path_slimmed(entry.path(), lazy)?;
                    if !dir.is_empty() {
                        current_dir.directories.push(dir);
                    }
                } else if entry_type.is_file() {
                    let file = LazyFile::new(entry.file_name().to_string_lossy().into_owned());

                    if !lazy {
                        file.load_content(path)?;
                    } else {
                        let mut reader = fs::File::open(entry.path())?;
                        file.set_is_text(&mut reader)?;
                    }

                    if file.get_is_text().unwrap_or(true) {
                        current_dir.files.push(file);
                    }
                }
            }
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Path is not a directory",
            ));
        }

        Ok(current_dir)
    }

    /// Recursively loads file contents for all text files in the tree.
    pub fn load_recursive<P: AsRef<Path>>(&self, parent_path: P) -> std::io::Result<()> {
        let path = self.path(parent_path.as_ref());
        for file in &self.files {
            file.load_content(&path)?;
        }
        for dir in &self.directories {
            dir.load_recursive(&path)?;
        }
        Ok(())
    }

    /// Loads contents for files directly in this directory (non-recursive).
    pub fn load_local_files<P: AsRef<Path>>(&self, parent_path: P) -> std::io::Result<()> {
        let path = self.path(parent_path);
        for file in &self.files {
            file.load_content(&path)?;
        }
        Ok(())
    }

    /// Bottom-up prunes binary/unloaded files and empty subdirectories.
    ///
    /// Returns `true` if at least one file or non-empty subdirectory remains.
    pub fn prune(&mut self, default: bool) -> bool {
        self.files.retain(|f| f.prune(default));
        self.directories.retain_mut(|dir| dir.prune(default));
        !self.is_empty()
    }

    /// Returns `true` when both `directories` and `files` are empty.
    pub fn is_empty(&self) -> bool {
        self.directories.is_empty() && self.files.is_empty()
    }

    /// Recursively sorts files and subdirectories alphabetically for deterministic output.
    pub fn sort(&mut self) {
        self.files.sort_by(|a, b| a.name.cmp(&b.name));
        self.directories.sort_by(|a, b| a.name.cmp(&b.name));
        for dir in &mut self.directories {
            dir.sort();
        }
    }
}

impl Serialize for Directory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;

        let len = self.files.len() + self.directories.len();
        if len == 0 {
            return Err(serde::ser::Error::custom(format!(
                "Directory '{}' is empty and cannot be serialized.",
                self.name
            )));
        }

        let mut map = serializer.serialize_map(Some(len))?;
        for file in &self.files {
            map.serialize_entry(&file.name, file)?;
        }
        for dir in &self.directories {
            map.serialize_entry(&dir.name, dir)?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(prefix: &str) -> std::io::Result<Self> {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "spodeian_export_dir_walk_{}_{}_{}",
                prefix,
                std::process::id(),
                id
            ));
            if path.exists() {
                let _ = fs::remove_dir_all(&path);
            }
            fs::create_dir_all(&path)?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_lazy_file_manual_init_and_uninitialized_error() {
        let uninit = LazyFile::new("uninit.txt".to_string());
        assert!(serde_json::to_string(&uninit).is_err());

        let file = LazyFile::new("test.txt".to_string());
        file.set_content(Some("Hello World".to_string())).unwrap();
        let json = serde_json::to_string(&file).unwrap();
        assert_eq!(json, "\"Hello World\"");
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_directory_flattened_serialization_and_sorting() {
        let mut root = Directory::new("root".to_string());
        let f1 = LazyFile::new("z.txt".to_string());
        f1.set_content(Some("content z".to_string())).unwrap();

        let mut sub = Directory::new("sub".to_string());
        let f2 = LazyFile::new("a.txt".to_string());
        f2.set_content(Some("content a".to_string())).unwrap();
        sub.files.push(f2);

        root.files.push(f1);
        root.directories.push(sub);
        root.sort();

        let json = serde_json::to_string(&root).unwrap();
        assert!(json.contains("\"z.txt\":\"content z\""));
        assert!(json.contains("\"sub\":{\"a.txt\":\"content a\"}"));

        let empty = Directory::new("empty".to_string());
        assert!(serde_json::to_string(&empty).is_err());
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_slimmed_lazy_pipeline_and_binary_pruning() -> std::io::Result<()> {
        let tmp = TestDir::new("pipeline")?;
        let root_path = tmp.path();
        let src = root_path.join("src");
        let assets = src.join("assets");
        fs::create_dir_all(&assets)?;

        File::create(root_path.join("config.json"))?.write_all(b"{\"version\": \"1.0\"}")?;
        File::create(root_path.join("data.bin"))?.write_all(&[0x00, 0xDE, 0xAD, 0xBE, 0xEF])?;
        File::create(src.join("main.rs"))?.write_all(b"fn main() {}")?;
        File::create(assets.join("logo.png"))?
            .write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])?;

        let mut root_dir = Directory::from_path_slimmed(root_path, true)?;
        let config_file = root_dir
            .files
            .iter()
            .find(|f| f.name == "config.json")
            .unwrap();
        assert!(config_file.get_content().is_none());

        let base_parent = root_path.parent().unwrap();
        root_dir.load_recursive(base_parent)?;
        assert!(root_dir.prune(false));
        root_dir.sort();

        let json_val = serde_json::to_value(&root_dir).expect("Serialization failed");
        assert_eq!(json_val["config.json"], "{\"version\": \"1.0\"}");
        assert_eq!(json_val["src"]["main.rs"], "fn main() {}");
        assert!(json_val.get("data.bin").is_none());
        assert!(json_val["src"].get("assets").is_none());
        Ok(())
    }

    #[test]
    fn test_edge_cases_empty_large_and_missing() -> std::io::Result<()> {
        let tmp = TestDir::new("edge")?;

        // 0-byte file
        File::create(tmp.path().join("empty.txt"))?;
        let empty = LazyFile::new("empty.txt".to_string());
        empty.load_content(tmp.path())?;
        assert_eq!(empty.get_is_text(), Some(true));
        assert_eq!(empty.get_content(), Some(""));

        // Large file (> 1024 bytes)
        let mut big_data = vec![b'a'; 2048];
        big_data.extend_from_slice(b"end_of_file");
        File::create(tmp.path().join("large.txt"))?.write_all(&big_data)?;
        let large = LazyFile::new("large.txt".to_string());
        large.load_content(tmp.path())?;
        let loaded = large.get_content().unwrap();
        assert!(loaded.contains("end_of_file"));
        assert_eq!(loaded.len(), 2048 + 11);

        // Missing file returns NotFound
        let ghost = LazyFile::new("ghost.txt".to_string());
        let err = ghost.load_content(tmp.path()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);

        // Nested empty dirs prune to false
        let deep = tmp.path().join("level1/level2/level3");
        fs::create_dir_all(&deep)?;
        let tmp_empty = TestDir::new("empty_nested")?;
        fs::create_dir_all(tmp_empty.path().join("level1/level2/level3"))?;
        let mut dir = Directory::from_path_slimmed(tmp_empty.path(), true)?;
        assert!(!dir.prune(false));
        Ok(())
    }
}
