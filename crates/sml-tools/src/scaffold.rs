//! `aix scaffold <topic>`: creates contract-conforming stubs for a curriculum topic in every track
//! (never overwriting existing files), then runs `sync` to register the Rust modules.

use std::fs;

use serde_json::{json, Value};

use crate::curriculum::{Readme, Topic, README};
use crate::repo::{file_name, parent, relative, strip_num, Repo, TRACKS};
use crate::sync;

fn src_lines(text: &str) -> Value {
    let lines: Vec<String> = text.split_inclusive('\n').map(str::to_string).collect();
    let mut lines = lines;
    if let Some(last) = lines.last_mut() {
        if last.ends_with('\n') {
            last.pop();
        }
    }
    json!(lines)
}

fn link(from: &str, topic: &Topic, track: &str) -> String {
    relative(parent(from), &topic.path(track))
}

fn neighbours<'a>(readme: &'a Readme, topic: &Topic) -> (Option<&'a Topic>, Option<&'a Topic>) {
    let mut same: Vec<&Topic> = readme
        .topics
        .iter()
        .filter(|t| t.category() == topic.category())
        .collect();
    same.sort_by(|a, b| a.id.cmp(&b.id));
    let i = same.iter().position(|t| t.id == topic.id);
    match i {
        Some(i) => (i.checked_sub(1).map(|j| same[j]), same.get(i + 1).copied()),
        None => (None, None),
    }
}

fn track_table(from: &str, topic: &Topic) -> String {
    let labels = [
        ("markdown", "Architectural Reference (Markdown)"),
        ("pytorch2", "PyTorch 2.x Module"),
        ("jax", "Functional JAX Module"),
        ("candle", "Native Rust Candle Track"),
        ("kernels", "Hardware Kernel (CPU / WGSL)"),
        ("weights", "Golden SafeTensors"),
        ("notebooks", "Concept Notebook"),
    ];
    labels
        .iter()
        .filter(|(t, _)| topic.path(t) != from)
        .map(|(t, label)| {
            format!(
                "- **{label}:** [{}]({})",
                file_name(&topic.path(t)),
                link(from, topic, t)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn notebook(readme: &Readme, topic: &Topic) -> String {
    let from = topic.path("notebooks");
    let (prev, next) = neighbours(readme, topic);
    let nav = |t: Option<&Topic>, label: &str| match t {
        Some(t) => format!(
            "[{label}: {}. {}]({})",
            t.num(),
            t.title,
            link(&from, t, "notebooks")
        ),
        None => format!("{label}: —"),
    };
    let index = format!("{}#{}", relative(parent(&from), README), topic.category());
    let where_ = match topic.subfolder() {
        Some(s) => format!("{} / {s}", topic.category()),
        None => topic.category().to_string(),
    };
    let up = prev
        .map(|t| format!("[{}]({})", t.title, link(&from, t, "notebooks")))
        .unwrap_or_else(|| "TODO".into());
    let down = next
        .map(|t| format!("[{}]({})", t.title, link(&from, t, "notebooks")))
        .unwrap_or_else(|| "TODO".into());
    let id = &topic.id;
    let cells: Vec<(&str, String)> = vec![
        ("markdown", format!(
            "# {}. {}\n\n- **Category:** {where_}\n- **Lineage & Citations:** TODO (authors, year, *paper title*).\n- **Purpose:** TODO\n- **Topic Connections:** `🔗 Upstream Prerequisites:` {up} $\\to$ `🔗 Downstream Syntheses:` {down}\n\n> [!WARNING]\n> **Decision Boundary: When NOT to use it**  \n> TODO",
            topic.num(), topic.title)),
        ("markdown", "## Mathematical Foundations & Governing Equations\n\n$$\n\\text{TODO}\n$$\n\n### Asymptotic Time & Memory Complexity\n- TODO: $\\mathcal{O}(\\cdot)$ time and memory.".to_string()),
        ("code", format!(
            "import torch\nimport safetensors.torch\nfrom pathlib import Path\nimport core\n\nenv = core.setup_env(seed=42)\n\n# Locate the golden checkpoint from the repo root or the notebook's own folder.\nCKPT_REL = Path(\"weights/{id}.safetensors\")\nckpt_path = next(p / CKPT_REL for p in (Path.cwd(), *Path.cwd().parents) if (p / CKPT_REL).exists())\nweights = safetensors.torch.load_file(ckpt_path)\nprint(f\"Loaded {{len(weights)}} golden tensors on {{env.device}}\")")),
        ("code", "def linear_anchor(x, w, b):\n    return x @ w + b\n\n# TODO: first-principles implementation of the topic.".to_string()),
        ("code", "y = linear_anchor(weights[\"test_input\"], weights[\"weight\"], weights[\"bias\"])\nerr = float(torch.max(torch.abs(y - weights[\"test_output\"])))\nassert err < 1e-6, f\"Linear anchor parity error: {err:.2e}\"\n# TODO: topic-specific parity assertions.\nprint(f\"Parity verified: {err:.2e}\")".to_string()),
        ("code", "from core.benchmark import benchmark_fn, format_benchmark_table\n\nb_anchor = benchmark_fn(lambda: linear_anchor(weights[\"test_input\"], weights[\"weight\"], weights[\"bias\"]), iters=50, name=\"Linear anchor\")\nprint(format_benchmark_table([b_anchor]))\n# TODO: diagnostic visualisation.".to_string()),
        ("code", "# Pathological Misuse vs Corrected Implementation\n# TODO: demonstrate the failure mode programmatically.\nfailure_demonstrated = True\nassert failure_demonstrated, \"Failure mode must be demonstrated\"".to_string()),
        ("markdown", format!(
            "## Algorithmic Takeaways & Cross-Domain Isomorphisms\n\n### Key Takeaways\n1. TODO\n\n---\n### Cross-Domain Generalisation Matrix\n- TODO\n\n---\n### Synchronized Track Navigation\n{}\n\n---\n**Curriculum Navigation:**  \n&larr; {} | [Master Index]({index}) | {} &rarr;",
            track_table(&from, topic), nav(prev, "Previous"), nav(next, "Next"))),
    ];
    let cells: Vec<Value> = cells
        .into_iter()
        .enumerate()
        .map(|(i, (ty, text))| {
            let mut c = json!({"cell_type": ty, "id": format!("aix-cell-{}", i + 1), "metadata": {}, "source": src_lines(&text)});
            if ty == "code" {
                c["execution_count"] = Value::Null;
                c["outputs"] = json!([]);
            }
            c
        })
        .collect();
    let nb = json!({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    });
    serde_json::to_string_pretty(&nb).unwrap_or_default() + "\n"
}

fn markdown(readme: &Readme, topic: &Topic) -> String {
    let from = topic.path("markdown");
    let mut s = format!("# {}\n\n", topic.title);
    for (i, spec) in readme.contract.markdown.sections.iter().enumerate() {
        let name = spec.split('|').next().unwrap_or(spec).trim();
        s.push_str(&format!("## {}. {name}\n\n", i + 1));
        if name.to_lowercase().contains("navigation") {
            s.push_str(&track_table(&from, topic));
            s.push_str(&format!(
                "\n- **Master Index:** [README]({}#{})\n\n",
                relative(parent(&from), README),
                topic.category()
            ));
        } else {
            s.push_str("TODO\n\n");
        }
    }
    s
}

fn pytorch(topic: &Topic) -> String {
    let id = &topic.id;
    format!(
        r#""""{title} — PyTorch 2.x reference implementation.

Topic: {id}
Golden weights: weights/{id}.safetensors

``python pytorch2/{id}.py`` verifies against the golden checkpoint;
``python pytorch2/{id}.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "{id}"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{{TOPIC}}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {{
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }}
    # TODO: add topic-specific inputs, parameters and reference outputs.
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {{k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}}
    save_file(tensors, str(path), metadata={{"format": "ai-examples-v1", "seed": "42", "topic": TOPIC}})
    print(f"Saved {{len(tensors)}} tensors to {{path.relative_to(REPO_ROOT)}}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{{name}}: max abs error {{err:.2e}}"
    print(f"{{TOPIC}}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
"#,
        title = topic.title
    )
}

fn jax(topic: &Topic) -> String {
    let id = &topic.id;
    format!(
        r#""""{title} — functional JAX implementation.

Topic: {id}. Run ``python jax/{id}.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "{id}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


# TODO: topic-specific pure functions.


def verify_against_golden() -> None:
    g = {{k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {{err:.2e}}"
    print("{id}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
"#,
        title = topic.title
    )
}

fn candle(topic: &Topic) -> String {
    format!(
        "//! {}\n//! in candle/{}.\n\nuse candle_core::{{Result, Tensor}};\n\n/// Linear anchor for cross-track parity verification.\npub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {{\n    x.matmul(weight)?.broadcast_add(bias)\n}}\n\n// TODO: topic-specific operators.\n",
        topic.title,
        parent(&topic.id)
    )
}

fn kernel(topic: &Topic) -> String {
    format!(
        "//! CPU (and optional WGSL) kernels for {}.\n\n/// CPU linear anchor: `y[b, o] = bias[o] + sum_i x[b, i] * w[i, o]`.\npub fn linear_anchor_cpu(x: &[f32], w: &[f32], bias: &[f32], batch: usize, d_in: usize, d_out: usize) -> Vec<f32> {{\n    let mut y = vec![0.0f32; batch * d_out];\n    for (b, row) in y.chunks_mut(d_out).enumerate() {{\n        for (o, out) in row.iter_mut().enumerate() {{\n            *out = bias[o] + (0..d_in).map(|i| x[b * d_in + i] * w[i * d_out + o]).sum::<f32>();\n        }}\n    }}\n    y\n}}\n\n// TODO: fused WGSL compute shader + `*_auto` dispatch with CPU fallback.\n",
        topic.title
    )
}

fn candle_test(topic: &Topic, module: &str) -> String {
    let rel = topic.id.as_str();
    let fn_name = module.replacen('t', "test_", 1);
    format!(
        "use candle_core::{{safetensors::load, Device, Result}};\n\nuse ai_candle::c{cat}::{module} as topic;\n\n#[path = \"../common/mod.rs\"]\nmod common;\n\n#[test]\n#[ignore = \"TODO: remove once weights/{rel}.safetensors and the implementation exist\"]\nfn {fn_name}_parity() -> Result<()> {{\n    let device = Device::Cpu;\n    let w = load(common::get_weights_path(\"{rel}.safetensors\"), &device)?;\n    let y = topic::linear_anchor(&w[\"test_input\"], &w[\"weight\"], &w[\"bias\"])?;\n    common::assert_max_diff(&y, &w[\"test_output\"], 1e-5)\n}}\n",
        cat = topic.category()
    )
}

/// Creates missing stubs; returns repo-relative paths of created files.
pub fn scaffold(
    repo: &mut Repo,
    readme: &Readme,
    query: &str,
    dry_run: bool,
    force: bool,
) -> Result<Vec<String>, String> {
    let topic = readme.resolve(query).map_err(|e| {
        format!("{e}.\nAdd it to the README curriculum first, e.g.\n  - [ ] **NN. Title** — [`NN_slug`](notebooks/<category>[/<subfolder>]/NN_slug.ipynb)")
    })?.clone();
    let mut created = Vec::new();
    let root = repo.root.clone();
    let mut put = |rel: String, content: String| -> Result<(), String> {
        let abs = root.join(&rel);
        if abs.exists() && !force {
            return Ok(());
        }
        if !dry_run {
            fs::create_dir_all(abs.parent().expect("has parent"))
                .map_err(|e| format!("{rel}: {e}"))?;
            fs::write(&abs, content).map_err(|e| format!("{rel}: {e}"))?;
        }
        created.push(rel);
        Ok(())
    };
    for (track, _) in TRACKS {
        let rel = topic.path(track);
        let content = match track {
            "markdown" => markdown(readme, &topic),
            "notebooks" => notebook(readme, &topic),
            "pytorch2" => pytorch(&topic),
            "jax" => jax(&topic),
            "candle" => candle(&topic),
            "kernels" => kernel(&topic),
            _ => continue, // weights come from `python pytorch2/<id>.py --export-golden`
        };
        put(rel, content)?;
    }
    if dry_run {
        return Ok(created);
    }
    repo.reload();

    // Candle integration test stub, numbered like the generated module.
    let ids = sync::category_topics(repo, "candle", topic.category());
    if let Some(idx) = ids.iter().position(|i| *i == topic.id) {
        let module = sync::module_name(idx, &topic.id);
        let dir = format!("crates/ai-candle/tests/{}", topic.category());
        let stem_tail = strip_num(topic.slug());
        let exists = repo
            .files_with(&dir, "rs")
            .any(|f| file_name(f).contains(stem_tail));
        if !exists || force {
            put(
                format!("{dir}/{}.rs", module.replacen('t', "test_", 1)),
                candle_test(&topic, &module),
            )?;
        }
    }
    repo.reload();
    let readme = crate::curriculum::load(repo)?;
    sync::write(repo, &sync::generate(repo, &readme))?;
    Ok(created)
}

/// Returns the single active `[~]` WIP topic (if any), or the first `[ ]` planned topic in sequential order.
#[allow(dead_code)]
pub fn next_topic<'a>(
    repo: &Repo,
    readme: &'a Readme,
) -> Option<(&'a Topic, crate::curriculum::Status, Vec<&'static str>)> {
    next_topic_filtered(repo, readme, None)
}

/// Filtered next topic resolver by optional category prefix.
pub fn next_topic_filtered<'a>(
    repo: &Repo,
    readme: &'a Readme,
    cat_filter: Option<&str>,
) -> Option<(&'a Topic, crate::curriculum::Status, Vec<&'static str>)> {
    for t in &readme.topics {
        if let Some(cat) = cat_filter {
            if t.category() != cat && !t.category().starts_with(cat) {
                continue;
            }
        }
        let (s, missing) = sync::derived(repo, &t.id);
        if s == crate::curriculum::Status::Wip {
            return Some((t, s, missing));
        }
    }
    for t in &readme.topics {
        if let Some(cat) = cat_filter {
            if t.category() != cat && !t.category().starts_with(cat) {
                continue;
            }
        }
        let (s, missing) = sync::derived(repo, &t.id);
        if s == crate::curriculum::Status::Planned {
            return Some((t, s, missing));
        }
    }
    None
}

/// Parses `# %% [markdown]` and `# %%` percent-format source into `(cell_type, text)` pairs.
pub fn parse_percent_cells(raw: &str) -> Vec<(String, String)> {
    let mut cells: Vec<(String, Vec<String>)> = Vec::new();
    for line in raw.replace("\r\n", "\n").lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("# %%") {
            let ty = if rest.to_lowercase().contains("[markdown]")
                || rest.to_lowercase().contains("[md]")
            {
                "markdown"
            } else {
                "code"
            };
            cells.push((ty.to_string(), Vec::new()));
            continue;
        }
        if let Some((ty, buf)) = cells.last_mut() {
            if ty == "markdown" {
                if let Some(stripped) = line.strip_prefix("# ") {
                    buf.push(stripped.to_string());
                } else if line.trim() == "#" {
                    buf.push(String::new());
                } else {
                    buf.push(line.to_string());
                }
            } else {
                buf.push(line.to_string());
            }
        }
    }
    cells
        .into_iter()
        .map(|(ty, buf)| (ty, buf.join("\n").trim().to_string()))
        .collect()
}

/// Compiles a `# %% [markdown]` / `# %%` percent-format file into a stripped 8-cell `.ipynb`,
/// exports an existing `.ipynb` to percent-format, or checks contract compliance.
pub fn notebook_tool(
    repo: &Repo,
    readme: &Readme,
    query: &str,
    from_file: Option<&str>,
    export_file: Option<&str>,
    check_only: bool,
) -> Result<String, String> {
    let topic = readme.resolve(query)?;
    let nb_rel = topic.path("notebooks");
    let nb_abs = repo.abs(&nb_rel);

    if check_only {
        if !repo.exists(&nb_rel) {
            return Err(format!(
                "{nb_rel} does not exist. Run `cargo aix scaffold {query}` to create it."
            ));
        }
        let scope = crate::repo::Scope {
            prefixes: vec![nb_rel.clone()],
        };
        let diags = crate::checks::notebooks(repo, readme, &scope, true);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == crate::report::Level::Error)
            .collect();
        if !errors.is_empty() {
            let mut msg = format!(
                "[FAIL] {nb_rel} has {} contract violation(s):\n",
                errors.len()
            );
            for e in errors {
                msg.push_str(&format!("  - {}\n", e.msg));
            }
            return Err(msg);
        }
        return Ok(format!("[PASS] {nb_rel} satisfies all 8-cell layout, output stripping, and semantic contract rules."));
    }

    if let Some(out_path) = export_file {
        let raw = fs::read_to_string(&nb_abs).map_err(|e| format!("{nb_rel}: {e}"))?;
        let nb: Value = serde_json::from_str(&raw).map_err(|e| format!("{nb_rel}: {e}"))?;
        let cells = nb
            .get("cells")
            .and_then(Value::as_array)
            .ok_or("missing cells array")?;
        let mut s = String::new();
        for (i, c) in cells.iter().enumerate() {
            let ty = c.get("cell_type").and_then(Value::as_str).unwrap_or("code");
            let text = crate::checks::source_text(c);
            if i > 0 {
                s.push_str("\n\n");
            }
            if ty == "markdown" {
                s.push_str("# %% [markdown]\n");
                for line in text.lines() {
                    if line.is_empty() {
                        s.push_str("#\n");
                    } else {
                        s.push_str("# ");
                        s.push_str(line);
                        s.push('\n');
                    }
                }
            } else {
                s.push_str("# %%\n");
                s.push_str(text.trim_end());
                s.push('\n');
            }
        }
        if out_path == "-" {
            print!("{s}");
            return Ok(format!("exported {nb_rel} to stdout"));
        }
        fs::write(out_path, s).map_err(|e| format!("{out_path}: {e}"))?;
        return Ok(format!("exported {nb_rel} -> {out_path}"));
    }

    let Some(src_path) = from_file else {
        if !repo.exists(&nb_rel) {
            return Err(format!(
                "{nb_rel} does not exist. Run `cargo aix scaffold {query}` to create it."
            ));
        }
        let raw = fs::read_to_string(&nb_abs).map_err(|e| format!("{nb_rel}: {e}"))?;
        let nb: Value = serde_json::from_str(&raw).map_err(|e| format!("{nb_rel}: {e}"))?;
        let cells = nb
            .get("cells")
            .and_then(Value::as_array)
            .ok_or("missing cells array")?;
        let mut report = format!("Notebook: {nb_rel} ({} cells)\n", cells.len());
        let mut stripped = true;
        let mut total_lines = 0;
        for (i, c) in cells.iter().enumerate() {
            let ty = c
                .get("cell_type")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let text = crate::checks::source_text(c);
            let lines = text.lines().count();
            total_lines += lines;
            let outputs = c
                .get("outputs")
                .and_then(Value::as_array)
                .map(|a| a.len())
                .unwrap_or(0);
            let exec = c.get("execution_count");
            if outputs > 0 || (exec.is_some() && !exec.unwrap().is_null()) {
                stripped = false;
            }
            let first_line = text.lines().next().unwrap_or("").trim();
            let summary = if first_line.len() > 50 {
                format!("{}...", &first_line[..47])
            } else {
                first_line.to_string()
            };
            report.push_str(&format!(
                "  Cell {:>2}: [{:<8}] {:>3} lines | {}\n",
                i + 1,
                ty,
                lines,
                summary
            ));
        }
        report.push_str(&format!(
            "\nTotal lines: {total_lines} (contract max: {})\n",
            readme.contract.notebook.max_lines
        ));
        report.push_str(&format!(
            "Stripped:    {}\n",
            if stripped {
                "yes (0 outputs, execution_count null)"
            } else {
                "NO (outputs or execution counts present)"
            }
        ));
        report.push_str(&format!("\nCommands:\n  Export:  cargo aix notebook {query} --export <file.py>\n  Compile: cargo aix notebook {query} --from <file.py>"));
        return Ok(report);
    };
    let raw = fs::read_to_string(src_path).map_err(|e| format!("{src_path}: {e}"))?;
    let parsed = parse_percent_cells(&raw);
    let want_len = readme.contract.notebook.cells.len();
    if parsed.len() != want_len {
        return Err(format!(
            "expected {want_len} `# %%` cells in {src_path}, found {}",
            parsed.len()
        ));
    }
    for (i, ((got_ty, _), (want_ty, desc))) in parsed
        .iter()
        .zip(readme.contract.notebook.cells.iter())
        .enumerate()
    {
        if got_ty != want_ty {
            return Err(format!(
                "cell {} ({desc}): expected {want_ty}, got {got_ty}",
                i + 1
            ));
        }
    }
    let cells: Vec<Value> = parsed
        .into_iter()
        .enumerate()
        .map(|(i, (ty, text))| {
            let mut c = json!({
                "cell_type": ty,
                "id": format!("aix-cell-{}", i + 1),
                "metadata": {},
                "source": src_lines(&text),
            });
            if ty == "code" {
                c["execution_count"] = Value::Null;
                c["outputs"] = json!([]);
            }
            c
        })
        .collect();
    let nb = json!({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    });
    if let Some(parent_dir) = nb_abs.parent() {
        fs::create_dir_all(parent_dir).map_err(|e| format!("{nb_rel}: {e}"))?;
    }
    let out_json = serde_json::to_string_pretty(&nb).unwrap_or_default() + "\n";
    fs::write(&nb_abs, out_json).map_err(|e| format!("{nb_rel}: {e}"))?;
    Ok(format!(
        "compiled {src_path} -> {nb_rel} ({want_len} cells)"
    ))
}
