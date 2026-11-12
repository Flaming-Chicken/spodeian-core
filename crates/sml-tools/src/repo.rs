//! Repository discovery, file index and POSIX-style relative path utilities.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

/// The seven mirrored tracks and the file extension of a topic in each.
pub const TRACKS: [(&str, &str); 7] = [
    ("markdown", "md"),
    ("notebooks", "ipynb"),
    ("pytorch2", "py"),
    ("jax", "py"),
    ("candle", "rs"),
    ("kernels", "rs"),
    ("weights", "safetensors"),
];

pub const CATEGORIES: [&str; 18] = [
    "00_foundations",
    "01_paradigms_and_dichotomies",
    "02_classical_and_probabilistic_ml",
    "03_building_blocks_and_losses",
    "04_training_methods_and_optimisers",
    "05_architectures",
    "06_smooth_and_physics_informed_ai",
    "07_domain_tasks",
    "08_learning_paradigms_rl_and_alignment",
    "09_causal_ai_and_evolutionary_search",
    "10_xai_and_ai_analysis_methods",
    "11_ai_safety_privacy_and_verification",
    "12_compilation_and_performance",
    "13_precision_and_numerics",
    "14_quantisation",
    "15_compression_peft_and_speculative_inference",
    "16_distributed_and_scalable",
    "17_deployment_low_level_and_capstones",
];

pub fn track_ext(track: &str) -> Option<&'static str> {
    TRACKS.iter().find(|(t, _)| *t == track).map(|(_, e)| *e)
}

/// `NN_rest` file/dir names identify ordered curriculum items.
pub fn is_numbered(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() > 3 && b[0].is_ascii_digit() && b[1].is_ascii_digit() && b[2] == b'_'
}

/// Strips the `NN_` ordering prefix from a slug.
pub fn strip_num(slug: &str) -> &str {
    if is_numbered(slug) {
        &slug[3..]
    } else {
        slug
    }
}

pub struct Repo {
    pub root: PathBuf,
    pub files: BTreeSet<String>,
    pub dirs: BTreeSet<String>,
    lower: BTreeMap<String, String>,
}

impl Repo {
    /// Walks up from `start` to the workspace root (Cargo.toml with `[workspace]` + README.md).
    pub fn find_root(start: &Path) -> Result<PathBuf, String> {
        let mut cur = absolute(start);
        loop {
            let manifest = cur.join("Cargo.toml");
            if manifest.is_file()
                && cur.join("README.md").is_file()
                && fs::read_to_string(&manifest)
                    .map(|s| s.contains("[workspace]"))
                    .unwrap_or(false)
            {
                return Ok(cur);
            }
            if !cur.pop() {
                return Err(format!(
                    "could not find the AI-Examples workspace root above {}",
                    start.display()
                ));
            }
        }
    }

    pub fn load(root: PathBuf) -> Repo {
        let files = git_files(&root).unwrap_or_else(|| {
            let mut out = BTreeSet::new();
            walk(&root, "", &mut out);
            out
        });
        let mut dirs = BTreeSet::new();
        for f in &files {
            let mut p = f.as_str();
            while let Some(i) = p.rfind('/') {
                p = &p[..i];
                if !dirs.insert(p.to_string()) {
                    break;
                }
            }
        }
        let lower = files
            .iter()
            .chain(dirs.iter())
            .map(|p| (p.to_lowercase(), p.clone()))
            .collect();
        Repo {
            root,
            files,
            dirs,
            lower,
        }
    }

    pub fn reload(&mut self) {
        *self = Repo::load(self.root.clone());
    }

    pub fn exists(&self, rel: &str) -> bool {
        rel.is_empty() || self.files.contains(rel) || self.dirs.contains(rel)
    }

    /// Existing path that differs from `rel` only by case (would break on case-sensitive filesystems).
    pub fn case_variant(&self, rel: &str) -> Option<&String> {
        self.lower
            .get(&rel.to_lowercase())
            .filter(|v| v.as_str() != rel)
    }

    pub fn abs(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    pub fn read(&self, rel: &str) -> Result<String, String> {
        fs::read_to_string(self.abs(rel)).map_err(|e| format!("{rel}: {e}"))
    }

    /// Files under `prefix/` with the given extension.
    pub fn files_with<'a>(
        &'a self,
        prefix: &'a str,
        ext: &'a str,
    ) -> impl Iterator<Item = &'a String> + 'a {
        let dir = format!("{prefix}/");
        let suffix = format!(".{ext}");
        self.files
            .range(dir.clone()..)
            .take_while(move |f| f.starts_with(&dir))
            .filter(move |f| f.ends_with(&suffix))
    }

    /// Topic ids (`<category>[/<subfolder>]/<NN_slug>`) present in a track.
    pub fn track_topics(&self, track: &str) -> BTreeSet<String> {
        let ext = track_ext(track).unwrap_or("");
        self.files_with(track, ext)
            .filter(|f| is_numbered(file_name(f)))
            .map(|f| f[track.len() + 1..f.len() - ext.len() - 1].to_string())
            .collect()
    }
}

fn git_files(root: &Path) -> Option<BTreeSet<String>> {
    let out = Command::new("git")
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ])
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(
        text.split('\0')
            .filter(|p| !p.is_empty() && !p.split('/').any(|s| s.starts_with('.')))
            .filter(|p| root.join(p).is_file())
            .map(str::to_string)
            .collect(),
    )
}

const PRUNE: [&str; 7] = [
    "target",
    "node_modules",
    "__pycache__",
    "venv",
    "env",
    "build",
    "dist",
];

fn walk(root: &Path, rel: &str, out: &mut BTreeSet<String>) {
    let Ok(entries) = fs::read_dir(root.join(rel)) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || PRUNE.contains(&name.as_str()) || name == "scratch" {
            continue;
        }
        let child = if rel.is_empty() {
            name
        } else {
            format!("{rel}/{name}")
        };
        match entry.file_type() {
            Ok(t) if t.is_dir() => walk(root, &child, out),
            Ok(t) if t.is_file() => {
                out.insert(child);
            }
            _ => {}
        }
    }
}

pub fn absolute(p: &Path) -> PathBuf {
    let base = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    };
    let mut out = PathBuf::new();
    for c in base.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Converts a user-supplied path into a repo-relative POSIX path (`""` for the root).
pub fn to_rel(root: &Path, p: &Path) -> Result<String, String> {
    let abs = absolute(p).to_string_lossy().replace('\\', "/");
    let root_s = root.to_string_lossy().replace('\\', "/");
    let (a, r) = if cfg!(windows) {
        (abs.to_lowercase(), root_s.to_lowercase())
    } else {
        (abs.clone(), root_s.clone())
    };
    if a == r {
        return Ok(String::new());
    }
    let prefix = format!("{}/", r.trim_end_matches('/'));
    if a.starts_with(&prefix) {
        Ok(abs[prefix.len()..].trim_end_matches('/').to_string())
    } else {
        Err(format!("{} is outside the repository", p.display()))
    }
}

pub fn file_name(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

pub fn parent(rel: &str) -> &str {
    rel.rfind('/').map(|i| &rel[..i]).unwrap_or("")
}

/// Joins `href` onto repo-relative `dir`, resolving `.`/`..`. `None` if it escapes the root.
pub fn join_rel(dir: &str, href: &str) -> Option<String> {
    let mut parts: Vec<&str> = if href.starts_with('/') {
        Vec::new()
    } else {
        dir.split('/').filter(|s| !s.is_empty()).collect()
    };
    for seg in href.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            s => parts.push(s),
        }
    }
    Some(parts.join("/"))
}

/// POSIX relative path from directory `from_dir` to repo-relative `to`.
pub fn relative(from_dir: &str, to: &str) -> String {
    let from: Vec<&str> = from_dir.split('/').filter(|s| !s.is_empty()).collect();
    let to_parts: Vec<&str> = to.split('/').filter(|s| !s.is_empty()).collect();
    let common = from
        .iter()
        .zip(&to_parts)
        .take_while(|(a, b)| a == b)
        .count();
    let mut out: Vec<&str> = vec![".."; from.len() - common];
    out.extend(&to_parts[common..]);
    if out.is_empty() {
        ".".to_string()
    } else {
        out.join("/")
    }
}

/// Repo-relative path prefixes limiting which files a command looks at (empty = everything).
#[derive(Clone, Debug, Default)]
pub struct Scope {
    pub prefixes: Vec<String>,
}

impl Scope {
    pub fn includes(&self, rel: &str) -> bool {
        self.prefixes.is_empty()
            || self.prefixes.iter().any(|p| {
                p.is_empty()
                    || rel == p
                    || (rel.starts_with(p.as_str()) && rel[p.len()..].starts_with('/'))
            })
    }
}

/// Converts text to the line-ending style of `original` (CRLF if it contains any CRLF).
pub fn match_eol(original: &str, text: &str) -> String {
    let lf = text.replace("\r\n", "\n");
    if original.contains("\r\n") {
        lf.replace('\n', "\r\n")
    } else {
        lf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_round_trip() {
        assert_eq!(
            relative("notebooks/05_a/03_b", "markdown/05_a/03_b/x.md"),
            "../../../markdown/05_a/03_b/x.md"
        );
        assert_eq!(
            relative("notebooks/05_a", "notebooks/05_a/x.ipynb"),
            "x.ipynb"
        );
        assert_eq!(
            join_rel("notebooks/05_a/03_b", "../../../markdown/x.md").as_deref(),
            Some("markdown/x.md")
        );
        assert_eq!(join_rel("a", "../../x"), None);
    }

    #[test]
    fn scope_matching() {
        let s = Scope {
            prefixes: vec!["notebooks/05_architectures".into()],
        };
        assert!(s.includes("notebooks/05_architectures/x.ipynb"));
        assert!(!s.includes("notebooks/05_architectures_extra/x.ipynb"));
        assert!(Scope::default().includes("anything"));
    }

    #[test]
    fn numbering() {
        assert!(is_numbered("01_rnn"));
        assert!(!is_numbered("common"));
        assert_eq!(strip_num("12_rnn_lstm"), "rnn_lstm");
    }
}
