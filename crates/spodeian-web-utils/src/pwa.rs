//! Static-site post-processing helpers for Content Security Policy inline-script hashing
//! and deterministic build-ID service-worker (`sw.js`) precache population.

use base64::Engine as _;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Placeholder replaced in `_headers` with `'sha256-...'` hashes of inline `<script>` tags.
pub const CSP_PLACEHOLDER: &str = "__INLINE_SCRIPT_HASHES__";

/// Extracts Base64-encoded SHA-256 digests of every non-empty inline `<script>` body in `html`
/// (skipping external scripts that carry a `src=` attribute).
pub fn inline_script_hashes(html: &str) -> Vec<String> {
    let mut hashes = Vec::new();
    let mut rest = html;
    while let Some(open) = rest.find("<script") {
        let after_open = &rest[open..];
        let Some(tag_end) = after_open.find('>') else {
            break;
        };
        let tag = &after_open[..tag_end];
        let body_and_rest = &after_open[tag_end + 1..];
        let Some(close) = body_and_rest.find("</script>") else {
            break;
        };
        let body = &body_and_rest[..close];
        if !tag.contains(" src=") && !body.is_empty() {
            let digest = Sha256::digest(body.as_bytes());
            hashes.push(base64::engine::general_purpose::STANDARD.encode(digest));
        }
        rest = &body_and_rest[close + "</script>".len()..];
    }
    hashes
}

/// Maps a static output file under `site_root` to its normalized URL path for service-worker
/// precaching, or `None` if the file should not be precached (`sw.js`, `_headers`, `.ts`, `.txt`).
pub fn precache_url(site_root: &Path, file: &Path) -> Option<String> {
    let relative = file.strip_prefix(site_root).ok()?;
    let path = relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");

    if path == "sw.js" || path == "_headers" || path.ends_with(".ts") || path.ends_with(".txt") {
        return None;
    }

    Some(match path.as_str() {
        "index.html" => "/".to_string(),
        "404.html" => "/404.html".to_string(),
        other => match other.strip_suffix(".html") {
            Some(page) => {
                let normalized = page.strip_suffix("/index").unwrap_or(page);
                if normalized.is_empty() {
                    "/".to_string()
                } else {
                    format!("/{normalized}")
                }
            }
            None => format!("/{other}"),
        },
    })
}

fn site_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(site_files(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Computes a deterministic 12-character SHA-256 content hash over all precacheable assets under
/// `site_root` and rewrites `'__BUILD_ID__'` and `/*__PRECACHE_URLS__*/ []` inside `sw.js`.
///
/// Returns `Ok(None)` when `site_root/sw.js` does not exist.
pub fn populate_service_worker(site_root: &Path) -> std::io::Result<Option<String>> {
    let worker = site_root.join("sw.js");
    if !worker.exists() {
        return Ok(None);
    }

    let mut urls = Vec::new();
    let mut digest = Sha256::new();
    for file in site_files(site_root) {
        let Some(url) = precache_url(site_root, &file) else {
            continue;
        };
        digest.update(url.as_bytes());
        digest.update(std::fs::read(&file)?);
        urls.push(url);
    }

    let final_digest = digest.finalize();
    let mut full_hex = String::with_capacity(64);
    for byte in final_digest {
        let _ = write!(&mut full_hex, "{byte:02x}");
    }
    let build_id = full_hex[..12].to_string();

    let list = urls
        .iter()
        .map(|u| format!("'{u}'"))
        .collect::<Vec<_>>()
        .join(", ");

    let text = std::fs::read_to_string(&worker)?
        .replace("'__BUILD_ID__'", &format!("'{build_id}'"))
        .replace("/*__PRECACHE_URLS__*/ []", &format!("[{list}]"));
    std::fs::write(&worker, text)?;

    Ok(Some(build_id))
}

/// Finalizes a static SSG build directory: rewrites [`CSP_PLACEHOLDER`] in `_headers` using
/// [`inline_script_hashes`] across `pages`, and populates `sw.js` via [`populate_service_worker`].
pub fn finalise_static_site(site_root: &Path, pages: &[PathBuf]) -> std::io::Result<Option<String>> {
    let mut hashes = BTreeSet::new();
    for page in pages {
        let html = std::fs::read_to_string(page)?;
        hashes.extend(inline_script_hashes(&html));
    }

    let headers = site_root.join("_headers");
    if headers.exists() {
        let text = std::fs::read_to_string(&headers)?;
        if text.contains(CSP_PLACEHOLDER) {
            let sources: Vec<String> = hashes.iter().map(|h| format!("'sha256-{h}'")).collect();
            std::fs::write(&headers, text.replace(CSP_PLACEHOLDER, &sources.join(" ")))?;
        }
    }

    populate_service_worker(site_root)
}
