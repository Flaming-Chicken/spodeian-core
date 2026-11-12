//! `aix check`: structure, notebook/markdown contracts, links, forbidden calls and weights.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;

use serde_json::Value;

use crate::curriculum::{Readme, Status, README};
use crate::repo::{
    file_name, is_numbered, join_rel, parent, relative, track_ext, Repo, Scope, CATEGORIES, TRACKS,
};
use crate::report::{Diag, Level};
use crate::sync;

pub const ALL: [&str; 6] = [
    "structure",
    "notebooks",
    "markdown",
    "links",
    "forbidden",
    "weights",
];

// ---------------------------------------------------------------------------------------------
// structure
// ---------------------------------------------------------------------------------------------

pub fn structure(repo: &Repo, readme: &Readme) -> Vec<Diag> {
    const C: &str = "structure";
    let mut out: Vec<Diag> = readme
        .problems
        .iter()
        .map(|(l, m)| Diag::error(C, README, m.clone()).at(l + 1))
        .collect();

    let mut ids: BTreeMap<&str, usize> = BTreeMap::new();
    let mut slots: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for t in &readme.topics {
        if let Some(prev) = ids.insert(&t.id, t.line) {
            out.push(
                Diag::error(
                    C,
                    README,
                    format!("topic '{}' listed twice (also line {})", t.id, prev + 1),
                )
                .at(t.line + 1),
            );
        }
        slots.entry(t.slot()).or_default().push(&t.id);
        if !CATEGORIES.contains(&t.category()) {
            out.push(
                Diag::error(C, README, format!("unknown category '{}'", t.category()))
                    .at(t.line + 1),
            );
        }
    }
    for (slot, list) in &slots {
        if list.len() > 1 {
            out.push(Diag::error(
                C,
                README,
                format!(
                    "slot {slot} is claimed by several topics: {}",
                    list.join(", ")
                ),
            ));
        }
    }

    for (track, _) in TRACKS {
        for cat in CATEGORIES {
            let dir = format!("{track}/{cat}");
            if !repo.dirs.contains(&dir) && !repo.abs(&dir).is_dir() {
                out.push(Diag::error(
                    C,
                    format!("{dir}/"),
                    "missing category directory (add a .gitkeep)",
                ));
            }
        }
        for id in repo.track_topics(track) {
            if !ids.contains_key(id.as_str()) {
                let ext = track_ext(track).unwrap_or("");
                out.push(Diag::error(
                    C,
                    format!("{track}/{id}.{ext}"),
                    "topic file is not listed in the README curriculum (add a checklist entry)",
                ));
            }
        }
    }

    let mut status_drift = false;
    for t in &readme.topics {
        let (derived, missing) = sync::derived(repo, &t.id);
        if derived != t.status {
            status_drift = true;
            out.push(
                Diag::error(
                    C,
                    README,
                    format!(
                        "'{}' is marked {} but is {} on disk — run `cargo aix sync`",
                        t.id,
                        t.status.label(),
                        derived.label()
                    ),
                )
                .at(t.line + 1),
            );
        }
        if derived == Status::Wip {
            out.push(Diag::warn(
                C,
                t.path("notebooks"),
                format!("work in progress; missing tracks: {}", missing.join(", ")),
            ));
        }
    }

    for path in sync::stale(repo, &sync::generate(repo, readme)) {
        if path == README && status_drift {
            continue;
        }
        out.push(Diag::error(
            C,
            path,
            "generated content is out of date — run `cargo aix sync`",
        ));
    }

    // scripts/ hygiene check: only permanent entry points are allowed in scripts/*.py
    for script in repo.files_with("scripts", "py") {
        let name = file_name(script);
        if matches!(name, "verify_repo.py" | "aix_nb_worker.py") {
            continue;
        }
        if name == "make_weights_05_03_02.py" {
            let wip_id = "05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm";
            if sync::derived(repo, wip_id).0 == Status::Wip {
                continue;
            }
        }
        out.push(Diag::error(
            C,
            script.clone(),
            "transient generator script in scripts/ — use `cargo aix notebook` or `pytorch2/<id>.py --export-golden` instead",
        ));
    }
    out
}

// ---------------------------------------------------------------------------------------------
// notebooks
// ---------------------------------------------------------------------------------------------

pub fn source_text(cell: &Value) -> String {
    match cell.get("source") {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

fn source_lines(cell: &Value) -> usize {
    match cell.get("source") {
        Some(Value::Array(a)) => a.len(),
        Some(Value::String(s)) => s.lines().count(),
        _ => 0,
    }
}

fn forbidden_hits<'a>(text: &str, patterns: &'a [String]) -> Vec<&'a str> {
    patterns
        .iter()
        .filter(|p| text.contains(p.as_str()))
        .map(String::as_str)
        .collect()
}

pub fn notebooks(repo: &Repo, readme: &Readme, scope: &Scope, semantics: bool) -> Vec<Diag> {
    const C: &str = "notebooks";
    let nc = &readme.contract.notebook;
    let mut out = Vec::new();
    let empty_cells = Vec::new();
    for rel in repo
        .files_with("notebooks", "ipynb")
        .filter(|f| scope.includes(f))
    {
        let nb: Value = match repo
            .read(rel)
            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
        {
            Ok(v) => v,
            Err(e) => {
                out.push(Diag::error(
                    C,
                    rel.clone(),
                    format!("invalid notebook JSON: {e}"),
                ));
                continue;
            }
        };
        if nb.get("nbformat").and_then(Value::as_u64) != Some(4) {
            out.push(Diag::error(
                C,
                rel.clone(),
                format!(
                    "expected nbformat 4, got {}",
                    nb.get("nbformat").unwrap_or(&Value::Null)
                ),
            ));
        }
        let cells = nb
            .get("cells")
            .and_then(Value::as_array)
            .unwrap_or(&empty_cells);
        if cells.len() != nc.cells.len() {
            out.push(Diag::error(
                C,
                rel.clone(),
                format!(
                    "expected exactly {} cells, found {}",
                    nc.cells.len(),
                    cells.len()
                ),
            ));
        }
        let mut total = 0usize;
        for (i, cell) in cells.iter().enumerate() {
            total += source_lines(cell);
            let ty = cell.get("cell_type").and_then(Value::as_str).unwrap_or("?");
            if let Some((want, desc)) = nc.cells.get(i) {
                if ty != want {
                    out.push(Diag::error(
                        C,
                        rel.clone(),
                        format!("cell {} ({desc}): expected {want}, got {ty}", i + 1),
                    ));
                }
            }
            if cell.get("id").is_none() {
                out.push(Diag::error(
                    C,
                    rel.clone(),
                    format!("cell {} is missing its 'id'", i + 1),
                ));
            }
            if ty == "code" {
                let text = source_text(cell);
                for p in forbidden_hits(&text, &readme.contract.forbidden_calls) {
                    out.push(Diag::error(
                        C,
                        rel.clone(),
                        format!("cell {} contains forbidden network call '{p}'", i + 1),
                    ));
                }
            }
        }
        if total > nc.max_lines {
            out.push(Diag::error(
                C,
                rel.clone(),
                format!(
                    "{total} source lines exceeds the {}-line ceiling",
                    nc.max_lines
                ),
            ));
        }
        if semantics && cells.len() == nc.cells.len() {
            let mut cell_texts: Vec<Option<String>> = vec![None; cells.len()];
            for rule in &nc.semantics {
                let cell_idx = rule.cell.saturating_sub(1);
                if cell_idx >= cells.len() {
                    continue;
                }
                let text =
                    cell_texts[cell_idx].get_or_insert_with(|| source_text(&cells[cell_idx]));
                if !rule.any.iter().any(|k| text.contains(k.as_str())) {
                    out.push(Diag::error(
                        C,
                        rel.clone(),
                        format!("cell {}: {}", rule.cell, rule.message),
                    ));
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// markdown
// ---------------------------------------------------------------------------------------------

pub fn markdown(repo: &Repo, readme: &Readme, scope: &Scope) -> Vec<Diag> {
    const C: &str = "markdown";
    let mc = &readme.contract.markdown;
    let level = if mc.enforce {
        Level::Error
    } else {
        Level::Warning
    };
    let mut out = Vec::new();
    for rel in repo
        .files_with("markdown", "md")
        .filter(|f| scope.includes(f) && is_numbered(file_name(f)))
    {
        let Ok(text) = repo.read(rel) else { continue };
        let mut fenced = false;
        let headings: Vec<String> = text
            .lines()
            .filter(|l| {
                if l.trim_start().starts_with("```") {
                    fenced = !fenced;
                }
                !fenced && l.starts_with("## ")
            })
            .map(|l| l.to_lowercase())
            .collect();
        let missing: Vec<&str> = mc
            .sections
            .iter()
            .filter(|spec| {
                !spec.split('|').any(|alt| {
                    headings
                        .iter()
                        .any(|h| h.contains(&alt.trim().to_lowercase()))
                })
            })
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            let mut d = Diag::error(
                C,
                rel.clone(),
                format!("missing required sections: {}", missing.join(", ")),
            );
            d.level = level;
            out.push(d);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// links
// ---------------------------------------------------------------------------------------------

/// Extracts inline link targets `[text](href)` / `![alt](src)` with 1-based line numbers,
/// skipping fenced code blocks and inline code spans.
pub fn extract_links(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut fenced = false;
    for (n, line) in text.lines().enumerate() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        let mut in_code = false;
        let cleaned: String = line
            .chars()
            .map(|c| {
                if c == '`' {
                    in_code = !in_code;
                    ' '
                } else if in_code {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        let bytes = cleaned.as_bytes();
        let mut i = 0;
        while let Some(pos) = cleaned[i..].find("](") {
            let start = i + pos + 2;
            let mut j = start;
            let mut depth = 0i32;
            let href = if bytes.get(start) == Some(&b'<') {
                match cleaned[start..].find('>') {
                    Some(e) => cleaned[start + 1..start + e].to_string(),
                    None => String::new(),
                }
            } else {
                while j < bytes.len() {
                    match bytes[j] {
                        b'(' => depth += 1,
                        b')' if depth == 0 => break,
                        b')' => depth -= 1,
                        b' ' | b'\t' => break,
                        _ => {}
                    }
                    j += 1;
                }
                cleaned[start..j].to_string()
            };
            if !href.is_empty() {
                out.push((n + 1, href));
            }
            i = start.max(j);
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn is_external(href: &str) -> bool {
    href.contains("://")
        || href.starts_with('#')
        || ["mailto:", "conversation:", "file:", "data:", "tel:"]
            .iter()
            .any(|p| href.starts_with(p))
}

/// True when `rel` is a track file or directory of a topic listed in the curriculum
/// (decision: links to planned topics stay as links and pass).
fn is_planned_target(readme: &Readme, rel: &str) -> bool {
    let Some((track, rest)) = rel.split_once('/') else {
        return false;
    };
    let Some(ext) = track_ext(track) else {
        return false;
    };
    if let Some(id) = rest.strip_suffix(&format!(".{ext}")) {
        return readme.topic(id).is_some();
    }
    readme
        .topics
        .iter()
        .any(|t| t.category() == rest || parent(&t.id) == rest)
}

fn target_ok(repo: &Repo, readme: &Readme, rel: &str) -> bool {
    repo.exists(rel)
        || repo.dirs.contains(rel)
        || repo.abs(rel).is_file()
        || repo.abs(rel).is_dir()
        || is_planned_target(readme, rel)
}

/// Proposes a corrected href for a broken link, if one can be derived unambiguously.
fn suggest(repo: &Repo, readme: &Readme, src: &str, resolved: &str, path: &str) -> Option<String> {
    let base = parent(src);
    // 1. Per-category README index links point at the category anchor in the root README.
    if file_name(path) == "README.md" {
        let parts: Vec<&str> = src.split('/').collect();
        if parts.len() >= 3 && track_ext(parts[0]).is_some() && CATEGORIES.contains(&parts[1]) {
            return Some(format!("{}#{}", relative(base, README), parts[1]));
        }
    }
    // 2. Wrong depth: the path contains a track segment; re-anchor from the last one.
    let segs: Vec<&str> = resolved.split('/').collect();
    let mut candidate = None;
    if let Some(k) = segs.iter().rposition(|s| track_ext(s).is_some()) {
        let anchored = segs[k..].join("/");
        if target_ok(repo, readme, &anchored) {
            return Some(relative(base, &anchored));
        }
        candidate = Some(anchored);
    }
    // 3. Stale slug (or wrong depth inside a category subfolder): another curriculum topic owns the slot.
    if let Some(c) = candidate.as_deref() {
        if let Some((track, rest)) = c.split_once('/') {
            if let Some(ext) = track_ext(track) {
                if let Some(id) = rest.strip_suffix(&format!(".{ext}")) {
                    let name = file_name(id);
                    if is_numbered(name) {
                        let slot = format!("{}/{}", parent(id), &name[..2]);
                        let suffix = format!("/{slot}");
                        if let Some(t) = readme.topics.iter().find(|t| {
                            (t.slot() == slot || t.slot().ends_with(&suffix)) && t.id != id
                        }) {
                            return Some(relative(base, &t.path(track)));
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn links(repo: &Repo, readme: &Readme, scope: &Scope, fix: bool) -> Vec<Diag> {
    const C: &str = "links";
    let mut out = Vec::new();
    let mut sources: Vec<(String, Vec<(Option<usize>, usize, String)>)> = Vec::new();

    for rel in repo
        .files
        .iter()
        .filter(|f| f.ends_with(".md") && scope.includes(f))
    {
        let Ok(text) = repo.read(rel) else { continue };
        let links = extract_links(&text)
            .into_iter()
            .map(|(l, h)| (None, l, h))
            .collect();
        sources.push((rel.clone(), links));
    }
    for rel in repo
        .files_with("notebooks", "ipynb")
        .filter(|f| scope.includes(f))
    {
        let Ok(nb) = repo
            .read(rel)
            .and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string()))
        else {
            continue;
        };
        let mut links = Vec::new();
        for (ci, cell) in nb
            .get("cells")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if cell.get("cell_type").and_then(Value::as_str) == Some("markdown") {
                links.extend(
                    extract_links(&source_text(cell))
                        .into_iter()
                        .map(|(l, h)| (Some(ci + 1), l, h)),
                );
            }
        }
        sources.push((rel.clone(), links));
    }

    for (src, links) in sources {
        let base = parent(&src).to_string();
        let mut fixes: Vec<(String, String)> = Vec::new();
        for (cell, line, href) in links {
            if is_external(&href) {
                continue;
            }
            let (path, frag) = match href.find(['#', '?']) {
                Some(i) => (&href[..i], &href[i..]),
                None => (href.as_str(), ""),
            };
            if path.is_empty() {
                continue;
            }
            let decoded = percent_decode(path);
            let mk = |msg: String| {
                let d = Diag::error(
                    C,
                    src.clone(),
                    match cell {
                        Some(c) => format!("cell {c}: {msg}"),
                        None => msg,
                    },
                );
                if cell.is_none() {
                    d.at(line)
                } else {
                    d
                }
            };
            let Some(resolved) = join_rel(&base, &decoded) else {
                out.push(mk(format!("link '{href}' escapes the repository root")));
                continue;
            };
            let resolved = resolved.trim_end_matches('/').to_string();
            if target_ok(repo, readme, &resolved) {
                continue;
            }
            if let Some(v) = repo.case_variant(&resolved) {
                let new = format!("{}{frag}", relative(&base, v));
                out.push(
                    mk(format!(
                        "link '{href}' differs in case from '{v}' (breaks on Linux)"
                    ))
                    .with_fix(new.clone()),
                );
                fixes.push((href.clone(), new));
                continue;
            }
            match suggest(repo, readme, &src, &resolved, &decoded) {
                Some(new) => {
                    let new = if new.contains('#') {
                        new
                    } else {
                        format!("{new}{frag}")
                    };
                    out.push(mk(format!("broken link '{href}'")).with_fix(new.clone()));
                    fixes.push((href.clone(), new));
                }
                None => out.push(mk(format!(
                    "broken link '{href}' (target does not exist and is not a curriculum topic)"
                ))),
            }
        }
        if fix && !fixes.is_empty() {
            if let Ok(raw) = fs::read_to_string(repo.abs(&src)) {
                let mut text = raw.clone();
                for (old, new) in &fixes {
                    text = text.replace(&format!("]({old})"), &format!("]({new})"));
                }
                if text != raw && fs::write(repo.abs(&src), &text).is_ok() {
                    for d in out
                        .iter_mut()
                        .filter(|d| d.file == src && d.fix.is_some() && d.level == Level::Error)
                    {
                        d.level = Level::Warning;
                        d.msg = format!("fixed: {}", d.msg);
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// forbidden calls in implementation tracks
// ---------------------------------------------------------------------------------------------

pub fn forbidden(repo: &Repo, readme: &Readme, scope: &Scope) -> Vec<Diag> {
    const C: &str = "forbidden";
    let mut out = Vec::new();
    let targets = [
        ("pytorch2", "py"),
        ("jax", "py"),
        ("candle", "rs"),
        ("kernels", "rs"),
        ("kernels", "py"),
        ("core", "py"),
    ];
    let mut seen = BTreeSet::new();
    for (dir, ext) in targets {
        for rel in repo.files_with(dir, ext).filter(|f| scope.includes(f)) {
            if !seen.insert(rel.clone()) {
                continue;
            }
            let Ok(text) = repo.read(rel) else { continue };
            for (n, line) in text.lines().enumerate() {
                for p in forbidden_hits(line, &readme.contract.forbidden_calls) {
                    out.push(
                        Diag::error(C, rel.clone(), format!("forbidden network call '{p}'"))
                            .at(n + 1),
                    );
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// weights (safetensors headers)
// ---------------------------------------------------------------------------------------------

pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset_start: usize,
    pub bytes: usize,
}

pub struct Header {
    pub data_offset: usize,
    pub tensors: Vec<TensorInfo>,
    pub metadata: BTreeMap<String, String>,
}

fn dtype_size(d: &str) -> Option<usize> {
    Some(match d {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return None,
    })
}

/// Reads and validates a safetensors header without loading tensor data.
pub fn read_header(path: &std::path::Path) -> Result<Header, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let file_len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .map_err(|_| "file shorter than 8 bytes".to_string())?;
    let n = u64::from_le_bytes(len_buf);
    if n > 100 << 20 || n + 8 > file_len {
        return Err(format!(
            "implausible header length {n} for a {file_len}-byte file"
        ));
    }
    let mut hdr = vec![0u8; n as usize];
    f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_slice(&hdr).map_err(|e| format!("header is not JSON: {e}"))?;
    let obj = v.as_object().ok_or("header is not a JSON object")?;
    let data_len = (file_len - 8 - n) as usize;
    let mut tensors = Vec::new();
    let mut metadata = BTreeMap::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            if let Some(m) = info.as_object() {
                metadata = m
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect();
            }
            continue;
        }
        let dtype = info
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or(format!("{name}: missing dtype"))?
            .to_string();
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or(format!("{name}: missing shape"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|x| x as usize)
                    .ok_or(format!("{name}: bad shape"))
            })
            .collect::<Result<_, _>>()?;
        let offs = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or(format!("{name}: missing data_offsets"))?;
        let (s, e) = match (
            offs.first().and_then(Value::as_u64),
            offs.get(1).and_then(Value::as_u64),
        ) {
            (Some(s), Some(e)) if s <= e => (s as usize, e as usize),
            _ => return Err(format!("{name}: bad data_offsets")),
        };
        if e > data_len {
            return Err(format!(
                "{name}: data_offsets end {e} beyond data section ({data_len} bytes)"
            ));
        }
        let size = dtype_size(&dtype).ok_or(format!("{name}: unknown dtype {dtype}"))?;
        let numel: usize = shape.iter().product();
        if numel * size != e - s {
            return Err(format!(
                "{name}: {numel} x {dtype} needs {} bytes but offsets span {}",
                numel * size,
                e - s
            ));
        }
        tensors.push(TensorInfo {
            name: name.clone(),
            dtype,
            shape,
            offset_start: s,
            bytes: e - s,
        });
    }
    Ok(Header {
        data_offset: 8 + n as usize,
        tensors,
        metadata,
    })
}

/// Compares two safetensors files tensor-by-tensor (shapes, dtypes, and max abs difference).
pub fn diff_safetensors(
    path_a: &std::path::Path,
    path_b: &std::path::Path,
) -> Result<bool, String> {
    let ha = read_header(path_a)?;
    let hb = read_header(path_b)?;
    let raw_a = fs::read(path_a).map_err(|e| e.to_string())?;
    let raw_b = fs::read(path_b).map_err(|e| e.to_string())?;
    let map_a: BTreeMap<&str, &TensorInfo> =
        ha.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let map_b: BTreeMap<&str, &TensorInfo> =
        hb.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let all_keys: BTreeSet<&str> = map_a.keys().chain(map_b.keys()).copied().collect();
    let mut identical = true;
    for k in all_keys {
        match (map_a.get(k), map_b.get(k)) {
            (Some(ta), None) => {
                println!("  - {k}: only in A ({} {:?})", ta.dtype, ta.shape);
                identical = false;
            }
            (None, Some(tb)) => {
                println!("  + {k}: only in B ({} {:?})", tb.dtype, tb.shape);
                identical = false;
            }
            (Some(ta), Some(tb)) => {
                if ta.dtype != tb.dtype || ta.shape != tb.shape {
                    println!(
                        "  ! {k}: shape/dtype mismatch ({} {:?} vs {} {:?})",
                        ta.dtype, ta.shape, tb.dtype, tb.shape
                    );
                    identical = false;
                    continue;
                }
                let slice_a = &raw_a
                    [ha.data_offset + ta.offset_start..ha.data_offset + ta.offset_start + ta.bytes];
                let slice_b = &raw_b
                    [hb.data_offset + tb.offset_start..hb.data_offset + tb.offset_start + tb.bytes];
                if slice_a == slice_b {
                    println!(
                        "  = {k:<28} {:<4} {:?} (byte-identical)",
                        ta.dtype, ta.shape
                    );
                } else if ta.dtype == "F32" {
                    let max_diff = slice_a
                        .chunks_exact(4)
                        .zip(slice_b.chunks_exact(4))
                        .map(|(ca, cb)| {
                            let va = f32::from_le_bytes([ca[0], ca[1], ca[2], ca[3]]);
                            let vb = f32::from_le_bytes([cb[0], cb[1], cb[2], cb[3]]);
                            (va - vb).abs()
                        })
                        .fold(0.0f32, f32::max);
                    println!(
                        "  ~ {k:<28} {:<4} {:?} (max_abs_diff = {max_diff:.3e})",
                        ta.dtype, ta.shape
                    );
                    if max_diff > 1e-5 {
                        identical = false;
                    }
                } else {
                    println!("  ! {k:<28} {:<4} {:?} (bytes differ)", ta.dtype, ta.shape);
                    identical = false;
                }
            }
            (None, None) => unreachable!(),
        }
    }
    Ok(identical)
}

pub fn weights(repo: &Repo, readme: &Readme, scope: &Scope) -> Vec<Diag> {
    const C: &str = "weights";
    let mut out = Vec::new();
    for rel in repo
        .files_with("weights", "safetensors")
        .filter(|f| scope.includes(f))
    {
        match read_header(&repo.abs(rel)) {
            Err(e) => out.push(Diag::error(
                C,
                rel.clone(),
                format!("invalid safetensors: {e}"),
            )),
            Ok(h) => {
                let names: BTreeSet<&str> = h.tensors.iter().map(|t| t.name.as_str()).collect();
                for alts in &readme.contract.weights.required_tensors {
                    if !alts.iter().any(|a| names.contains(a.as_str())) {
                        out.push(Diag::error(
                            C,
                            rel.clone(),
                            format!("missing required tensor {}", alts.join(" | ")),
                        ));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_extraction() {
        let text = "See [a](x.md) and ![img](p (1).png) `[no](code.md)`\n```\n[no](fenced.md)\n```\n[t](y.md \"title\")";
        let hrefs: Vec<String> = extract_links(text).into_iter().map(|(_, h)| h).collect();
        assert_eq!(hrefs, vec!["x.md", "p", "y.md"]);
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(percent_decode("a%20b.md"), "a b.md");
    }
}
