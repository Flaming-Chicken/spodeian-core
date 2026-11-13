//! `aix` — AI-Examples repository toolchain.
//!
//! Run `cargo aix help` (or `python scripts/verify_repo.py help`) for usage. README.md is the
//! shared planning document and the source of truth for the curriculum and the contracts.

mod checks;
mod curriculum;
mod help;
mod repo;
mod report;
mod run;
mod scaffold;
mod sync;

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use help::{USAGE, print_cmd_help};
use repo::{Repo, Scope};
use report::{Format, Section};

/// Minimal argument parser: positionals + `--flag` / `--key value` / `--key=value` + short flags.
struct Args {
    pos: Vec<String>,
    opts: BTreeMap<String, Vec<String>>,
}

impl Args {
    fn parse(argv: &[String], valued: &[&str], switches: &[&str]) -> Result<Args, String> {
        let mut pos = Vec::new();
        let mut opts: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut it = argv.iter();
        while let Some(a) = it.next() {
            if a == "--" {
                pos.extend(it.cloned());
                break;
            }
            if let Some(name) = a.strip_prefix("--") {
                let (key, inline) = match name.split_once('=') {
                    Some((k, v)) => (k, Some(v.to_string())),
                    None => (name, None),
                };
                if key == "help" {
                    opts.entry("help".to_string()).or_default();
                    continue;
                }
                if valued.contains(&key) {
                    let v = match inline {
                        Some(v) => v,
                        None => it.next().cloned().ok_or(format!("--{key} needs a value"))?,
                    };
                    opts.entry(key.to_string()).or_default().push(v);
                } else if switches.contains(&key) {
                    opts.entry(key.to_string()).or_default();
                } else {
                    return Err(format!("unknown option --{key}"));
                }
            } else if let Some(short) = a.strip_prefix('-') {
                if short.is_empty() {
                    pos.push(a.clone());
                    continue;
                }
                let mut chars = short.chars().peekable();
                while let Some(c) = chars.next() {
                    let key = match c {
                        'h' => "help",
                        'v' => "verbose",
                        'q' => "quiet",
                        'f' => "fix",
                        's' => "semantics",
                        'j' => "jobs",
                        'p' => "path",
                        't' => "topic",
                        'c' => "category",
                        _ => return Err(format!("unknown flag -{c}")),
                    };
                    if key == "help" {
                        opts.entry("help".to_string()).or_default();
                    } else if valued.contains(&key) {
                        let rest: String = chars.collect();
                        let v = if !rest.is_empty() {
                            rest
                        } else {
                            it.next().cloned().ok_or(format!("-{c} needs a value"))?
                        };
                        opts.entry(key.to_string()).or_default().push(v);
                        break;
                    } else if switches.contains(&key) {
                        opts.entry(key.to_string()).or_default();
                    } else {
                        return Err(format!("flag -{c} not supported for this command"));
                    }
                }
            } else {
                pos.push(a.clone());
            }
        }
        Ok(Args { pos, opts })
    }

    fn has(&self, k: &str) -> bool {
        self.opts.contains_key(k)
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.opts.get(k).and_then(|v| v.last()).map(String::as_str)
    }

    fn all(&self, k: &str) -> Vec<String> {
        self.opts.get(k).cloned().unwrap_or_default()
    }

    fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        self.get(k)
            .map(|v| {
                v.parse::<T>()
                    .map_err(|_| format!("--{k}: invalid number '{v}'"))
            })
            .transpose()
    }
}

struct Ctx {
    repo: Repo,
}

impl Ctx {
    fn readme(&self) -> Result<curriculum::Readme, String> {
        curriculum::load(&self.repo)
    }

    fn scope(
        &self,
        paths: &[String],
        topic: Option<&str>,
        readme: &curriculum::Readme,
    ) -> Result<Scope, String> {
        let mut prefixes = Vec::new();
        for p in paths {
            prefixes.push(repo::to_rel(&self.repo.root, Path::new(p))?);
        }
        if let Some(t) = topic {
            let t = readme.resolve(t)?;
            prefixes.extend(repo::TRACKS.iter().map(|(tr, _)| t.path(tr)));
        }
        Ok(Scope { prefixes })
    }
}

fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8)
}

struct CheckPlan {
    only: Vec<String>,
    scope: Scope,
    semantics: bool,
    fix: bool,
}

fn run_checks(
    ctx: &Ctx,
    readme: &curriculum::Readme,
    plan: &CheckPlan,
    format: Format,
    max: usize,
    quiet: bool,
) -> bool {
    let mut sections = Vec::new();
    for name in checks::ALL {
        if !plan.only.iter().any(|o| o == name) {
            continue;
        }
        let t = Instant::now();
        let (r, s) = (&ctx.repo, &plan.scope);
        let diags = match name {
            "structure" => checks::structure(r, readme),
            "notebooks" => checks::notebooks(r, readme, s, plan.semantics),
            "markdown" => checks::markdown(r, readme, s),
            "links" => checks::links(r, readme, s, plan.fix),
            "forbidden" => checks::forbidden(r, readme, s),
            "weights" => checks::weights(r, readme, s),
            _ => unreachable!(),
        };
        sections.push(Section {
            name,
            diags,
            seconds: t.elapsed().as_secs_f64(),
        });
    }
    report::emit(&sections, format, max, quiet)
}

fn cmd_check(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(
        argv,
        &["only", "path", "topic", "format", "max"],
        &["semantics", "fix", "quiet"],
    )?;
    if a.has("help") {
        print_cmd_help("check");
        return Ok(true);
    }
    let readme = ctx.readme()?;
    let only: Vec<String> = match a.get("only") {
        Some(list) => list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        None => checks::ALL.iter().map(|s| s.to_string()).collect(),
    };
    if let Some(bad) = only.iter().find(|o| !checks::ALL.contains(&o.as_str())) {
        return Err(format!(
            "unknown check '{bad}' (expected one of {})",
            checks::ALL.join(",")
        ));
    }
    let plan = CheckPlan {
        only,
        scope: ctx.scope(&a.all("path"), a.get("topic"), &readme)?,
        semantics: a.has("semantics"),
        fix: a.has("fix"),
    };
    let format = Format::parse(a.get("format").unwrap_or("human"))?;
    let t = Instant::now();
    let ok = run_checks(
        ctx,
        &readme,
        &plan,
        format,
        a.num("max")?.unwrap_or(25),
        a.has("quiet"),
    );
    if format != Format::Json {
        println!(
            "{} aix check finished in {:.2}s",
            if ok { "[PASS]" } else { "[FAIL]" },
            t.elapsed().as_secs_f64()
        );
    }
    Ok(ok)
}

fn cmd_sync(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &[], &["check"])?;
    if a.has("help") {
        print_cmd_help("sync");
        return Ok(true);
    }
    let readme = ctx.readme()?;
    let outputs = sync::generate(&ctx.repo, &readme);
    if a.has("check") {
        let stale = sync::stale(&ctx.repo, &outputs);
        for p in &stale {
            println!("[FAIL] out of date: {p}");
        }
        if stale.is_empty() {
            println!("[PASS] all generated content is up to date");
        } else {
            println!("Run `cargo aix sync` to regenerate.");
        }
        return Ok(stale.is_empty());
    }
    let changed = sync::write(&ctx.repo, &outputs)?;
    if changed.is_empty() {
        println!("Everything up to date.");
    }
    for p in changed {
        println!("updated {p}");
    }
    Ok(true)
}

fn cmd_status(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &["category", "format"], &["wip"])?;
    if a.has("help") {
        print_cmd_help("status");
        return Ok(true);
    }
    let cat_filter = a.get("category");
    let format = a.get("format").unwrap_or("human");
    let readme = ctx.readme()?;
    let mut per: BTreeMap<&str, [usize; 3]> = BTreeMap::new();
    let mut wip = Vec::new();
    let mut wip_details = Vec::new();
    for t in &readme.topics {
        if let Some(cat) = cat_filter
            && t.category() != cat
            && !t.category().starts_with(cat)
        {
            continue;
        }
        let (s, missing) = sync::derived(&ctx.repo, &t.id);
        per.entry(
            repo::CATEGORIES
                .iter()
                .find(|c| **c == t.category())
                .copied()
                .unwrap_or("?"),
        )
        .or_default()[s as usize] += 1;
        if s == curriculum::Status::Wip {
            wip.push(format!("  ~ {} (missing: {})", t.id, missing.join(", ")));
            wip_details.push(serde_json::json!({
                "id": t.id,
                "title": t.title,
                "category": t.category(),
                "missing": missing
            }));
        }
    }
    if format == "json" {
        let mut tot = [0usize; 3];
        let mut cat_map = serde_json::Map::new();
        for cat in repo::CATEGORIES {
            if let Some(cat_f) = cat_filter
                && cat != cat_f
                && !cat.starts_with(cat_f)
            {
                continue;
            }
            let c = per.get(cat).copied().unwrap_or_default();
            (0..3).for_each(|k| tot[k] += c[k]);
            cat_map.insert(
                cat.to_string(),
                serde_json::json!({
                    "done": c[2],
                    "wip": c[1],
                    "planned": c[0]
                }),
            );
        }
        let listed: usize = tot.iter().sum();
        let target = if cat_filter.is_some() {
            listed
        } else {
            readme.contract.target_topics.unwrap_or(listed).max(listed)
        };
        let out = serde_json::json!({
            "target": target,
            "listed": listed,
            "done": tot[2],
            "wip": tot[1],
            "planned": tot[0],
            "percentage": (1000.0 * tot[2] as f64 / target.max(1) as f64).round() / 10.0,
            "categories": cat_map,
            "wip_topics": wip_details
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        return Ok(true);
    }
    println!(
        "{:<48} {:>5} {:>5} {:>8}",
        "category", "done", "wip", "planned"
    );
    let mut tot = [0usize; 3];
    for cat in repo::CATEGORIES {
        if let Some(cat_f) = cat_filter
            && cat != cat_f
            && !cat.starts_with(cat_f)
        {
            continue;
        }
        let c = per.get(cat).copied().unwrap_or_default();
        (0..3).for_each(|k| tot[k] += c[k]);
        println!("{cat:<48} {:>5} {:>5} {:>8}", c[2], c[1], c[0]);
    }
    let listed: usize = tot.iter().sum();
    let target = if cat_filter.is_some() {
        listed
    } else {
        readme.contract.target_topics.unwrap_or(listed).max(listed)
    };
    println!("{:<48} {:>5} {:>5} {:>8}", "TOTAL", tot[2], tot[1], tot[0]);
    println!(
        "{} / {target} topics done ({:.0}%).",
        tot[2],
        100.0 * tot[2] as f64 / target.max(1) as f64
    );
    if a.has("wip") || wip.len() <= 10 {
        wip.iter().for_each(|w| println!("{w}"));
    }
    Ok(true)
}

fn nb_options(
    ctx: &Ctx,
    a: &Args,
    readme: &curriculum::Readme,
    scope: Scope,
) -> Result<run::NbOptions, String> {
    let _ = ctx;
    Ok(run::NbOptions {
        scope,
        jobs: a.num("jobs")?.unwrap_or_else(default_jobs),
        budget: a
            .num("budget")?
            .unwrap_or(readme.contract.notebook.budget_seconds),
        grace: a.num("grace")?.unwrap_or(45.0),
        python: a.get("python").map(str::to_string),
        verbose: a.has("verbose"),
    })
}

fn cmd_run_notebooks(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(
        argv,
        &["path", "topic", "jobs", "budget", "grace", "python"],
        &["verbose"],
    )?;
    if a.has("help") {
        print_cmd_help("run-notebooks");
        return Ok(true);
    }
    let readme = ctx.readme()?;
    let scope = ctx.scope(&a.all("path"), a.get("topic"), &readme)?;
    Ok(run::run_notebooks(
        &ctx.repo,
        &nb_options(ctx, &a, &readme, scope)?,
    ))
}

fn cmd_test(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(
        argv,
        &["topic", "crate", "jobs", "python", "budget", "grace"],
        &["verbose"],
    )?;
    if a.has("help") {
        print_cmd_help("test");
        return Ok(true);
    }
    let jobs: Option<usize> = a.num("jobs")?;
    let mut crates = a.all("crate");
    if crates.is_empty() {
        crates = vec!["sml-candle".into(), "sml-kernel".into()];
    }
    // `fold` rather than `all`: run every crate's tests even after a failure.
    let Some(q) = a.get("topic") else {
        #[allow(clippy::unnecessary_fold)]
        return Ok(crates.iter().fold(true, |ok, c| {
            run::cargo_test(&ctx.repo.root, c, None, jobs) && ok
        }));
    };
    let readme = ctx.readme()?;
    let topic = readme.resolve(q)?.clone();
    let filter = repo::strip_num(topic.slug()).to_string();
    #[allow(clippy::unnecessary_fold)]
    let mut ok = crates.iter().fold(true, |ok, c| {
        run::cargo_test(&ctx.repo.root, c, Some(&filter), jobs) && ok
    });
    let py = run::python(&ctx.repo, a.get("python"));
    let pt = topic.path("pytorch2");
    if ctx.repo.exists(&pt) {
        ok &= run::python_script(&ctx.repo, &py, &pt, &[]);
    }
    let jx = topic.path("jax");
    if ctx.repo.exists(&jx) {
        ok &= run::python_script(&ctx.repo, &py, &jx, &[]);
    }
    let scope = Scope {
        prefixes: vec![topic.path("notebooks")],
    };
    ok &= run::run_notebooks(&ctx.repo, &nb_options(ctx, &a, &readme, scope)?);
    let plan = CheckPlan {
        only: checks::ALL
            .iter()
            .filter(|c| **c != "structure")
            .map(|s| s.to_string())
            .collect(),
        scope: ctx.scope(&[], Some(&topic.id), &readme)?,
        semantics: true,
        fix: false,
    };
    ok &= run_checks(ctx, &readme, &plan, Format::Human, 50, false);
    println!(
        "{} topic {}",
        if ok { "[PASS]" } else { "[FAIL]" },
        topic.id
    );
    Ok(ok)
}

fn cmd_scaffold(ctx: &mut Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &[], &["dry-run", "force"])?;
    if a.has("help") {
        print_cmd_help("scaffold");
        return Ok(true);
    }
    let q = a
        .pos
        .first()
        .ok_or("usage: aix scaffold <topic> [--dry-run] [--force]")?;
    let readme = ctx.readme()?;
    let dry_run = a.has("dry-run");
    let force = a.has("force");
    let created = scaffold::scaffold(&mut ctx.repo, &readme, q, dry_run, force)?;
    if created.is_empty() {
        println!("Nothing to create; all track files already exist (use --force to overwrite).");
    }
    for c in &created {
        println!("{} {c}", if dry_run { "would create" } else { "created" });
    }
    if !dry_run && !created.is_empty() {
        println!(
            "Next: implement, then `python pytorch2/<id>.py --export-golden` and `cargo aix test --topic {q}`."
        );
    }
    Ok(true)
}

fn cmd_next(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &["category", "format"], &[])?;
    if a.has("help") {
        print_cmd_help("next");
        return Ok(true);
    }
    let cat_filter = a.get("category");
    let format = a.get("format").unwrap_or("human");
    let readme = ctx.readme()?;
    let result = scaffold::next_topic_filtered(&ctx.repo, &readme, cat_filter);
    if format == "json" {
        let json_val = match result {
            Some((t, s, missing)) => serde_json::json!({
                "status": s.label(),
                "topic": {
                    "id": t.id,
                    "title": t.title,
                    "category": t.category(),
                    "slug": t.slug(),
                    "missing_tracks": missing
                }
            }),
            None => serde_json::json!({
                "status": "complete",
                "message": "All curriculum topics in scope are complete!"
            }),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&json_val).unwrap_or_default()
        );
        return Ok(true);
    }
    match result {
        Some((t, curriculum::Status::Wip, missing)) => {
            println!("Active WIP topic [~]: {}", t.id);
            println!("  Title:          {}", t.title);
            println!("  Category:       {}", t.category());
            println!("  Missing tracks: {}", missing.join(", "));
            println!("  Next steps:");
            println!("    1. Complete missing tracks ({})", missing.join(", "));
            println!("    2. cargo aix weights export {}", t.id);
            println!("    3. cargo aix sync && cargo aix test --topic {}", t.id);
            Ok(true)
        }
        Some((t, _, _)) => {
            println!("Next planned topic [ ]: {}", t.id);
            println!("  Title:    {}", t.title);
            println!("  Category: {}", t.category());
            println!("  Start:    cargo aix scaffold {}", t.id);
            Ok(true)
        }
        None => {
            println!("All curriculum topics in scope are complete!");
            Ok(true)
        }
    }
}

fn cmd_notebook(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &["from", "export"], &["check"])?;
    if a.has("help") {
        print_cmd_help("notebook");
        return Ok(true);
    }
    let q = a
        .pos
        .first()
        .ok_or("usage: aix notebook <topic> [--from FILE | --export FILE | --check]")?;
    let readme = ctx.readme()?;
    let msg = scaffold::notebook_tool(
        &ctx.repo,
        &readme,
        q,
        a.get("from"),
        a.get("export"),
        a.has("check"),
    )?;
    println!("{msg}");
    Ok(true)
}

fn cmd_weights(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &["python", "topic", "path"], &["check"])?;
    if a.has("help") || a.pos.is_empty() {
        print_cmd_help("weights");
        return Ok(true);
    }
    match a.pos.first().map(String::as_str) {
        Some("inspect") if a.pos.len() > 1 => {
            let mut ok = true;
            for f in &a.pos[1..] {
                let path = repo::absolute(Path::new(f));
                match checks::read_header(&path) {
                    Ok(h) => {
                        println!("{f}");
                        for (k, v) in &h.metadata {
                            println!("  meta {k} = {v}");
                        }
                        let w = h.tensors.iter().map(|t| t.name.len()).max().unwrap_or(4);
                        for t in &h.tensors {
                            println!("  {:<w$}  {:<5} {:?} ({} B)", t.name, t.dtype, t.shape, t.bytes);
                        }
                    }
                    Err(e) => {
                        ok = false;
                        println!("[FAIL] {f}: {e}");
                    }
                }
            }
            Ok(ok)
        }
        Some("export") if a.pos.len() == 2 => {
            let readme = ctx.readme()?;
            let topic = readme.resolve(&a.pos[1])?;
            let script = topic.path("pytorch2");
            if !ctx.repo.exists(&script) {
                return Err(format!("{script} does not exist"));
            }
            let py = run::python(&ctx.repo, a.get("python"));
            Ok(run::python_script(&ctx.repo, &py, &script, &["--export-golden"]))
        }
        Some("diff") if a.pos.len() == 3 => {
            let pa = repo::absolute(Path::new(&a.pos[1]));
            let pb = repo::absolute(Path::new(&a.pos[2]));
            println!("Diffing {} vs {}:", a.pos[1], a.pos[2]);
            checks::diff_safetensors(&pa, &pb)
        }
        Some("check") => {
            let readme = ctx.readme()?;
            let scope = ctx.scope(&a.all("path"), a.get("topic"), &readme)?;
            let diags = checks::weights(&ctx.repo, &readme, &scope);
            let errors = diags.iter().filter(|d| d.level == report::Level::Error).count();
            for d in &diags {
                let tag = if d.level == report::Level::Error { "[FAIL]" } else { "[WARN]" };
                println!("{tag} {}: {}", d.file, d.msg);
            }
            if errors == 0 {
                println!("[PASS] all checked weights files are valid and contain required tensors");
            }
            Ok(errors == 0)
        }
        Some("regen") => {
            let readme = ctx.readme()?;
            let scope = ctx.scope(&a.all("path"), a.get("topic"), &readme)?;
            let py = run::python(&ctx.repo, a.get("python"));
            let mut ok = true;
            for t in &readme.topics {
                let pt = t.path("pytorch2");
                let wt = t.path("weights");
                if !ctx.repo.exists(&pt) || !scope.includes(&pt) {
                    continue;
                }
                let before = std::fs::read(ctx.repo.abs(&wt)).ok();
                if !run::python_script(&ctx.repo, &py, &pt, &["--export-golden"]) {
                    ok = false;
                    continue;
                }
                if a.has("check")
                    && let (Some(b), Ok(after)) = (before, std::fs::read(ctx.repo.abs(&wt)))
                        && b != after {
                            println!("[WARN] {wt} differed after --export-golden");
                        }
            }
            Ok(ok)
        }
        Some(topic_or_file) if a.pos.len() == 1 => {
            let path = if topic_or_file.ends_with(".safetensors") || Path::new(topic_or_file).is_file() {
                repo::absolute(Path::new(topic_or_file))
            } else {
                let readme = ctx.readme()?;
                let t = readme.resolve(topic_or_file)?;
                let wt = t.path("weights");
                if !ctx.repo.exists(&wt) {
                    return Err(format!("{wt} does not exist. Run `cargo aix weights export {}` to generate it.", t.id));
                }
                ctx.repo.abs(&wt)
            };
            let h = checks::read_header(&path)?;
            println!("{}", path.display());
            for (k, v) in &h.metadata {
                println!("  meta {k} = {v}");
            }
            let w = h.tensors.iter().map(|t| t.name.len()).max().unwrap_or(4);
            for t in &h.tensors {
                println!("  {:<w$}  {:<5} {:?} ({} B)", t.name, t.dtype, t.shape, t.bytes);
            }
            Ok(true)
        }
        _ => Err("usage: aix weights inspect FILE... | aix weights export <topic> | aix weights diff A B | aix weights check | aix weights regen | aix weights <topic_or_file>".into()),
    }
}

struct CleanEntry {
    path: std::path::PathBuf,
    rel: String,
    is_dir: bool,
}

fn walk_clean(dir: &Path, rel_prefix: &str, out: &mut Vec<CleanEntry>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let p = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str == ".git"
            || name_str == "target"
            || name_str == "node_modules"
            || name_str == ".venv"
            || name_str == "venv"
        {
            continue;
        }
        let rel = if rel_prefix.is_empty() {
            name_str.to_string()
        } else {
            format!("{rel_prefix}/{}", name_str)
        };
        if p.is_dir() {
            if name_str == "__pycache__" || name_str == ".ipynb_checkpoints" {
                out.push(CleanEntry {
                    path: p,
                    rel,
                    is_dir: true,
                });
            } else {
                walk_clean(&p, &rel, out);
            }
        } else if name_str.ends_with(".pyc") || name_str.ends_with(".pyo") {
            out.push(CleanEntry {
                path: p,
                rel,
                is_dir: false,
            });
        }
    }
}

fn cmd_clean(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &[], &["dry-run"])?;
    if a.has("help") {
        print_cmd_help("clean");
        return Ok(true);
    }
    let dry_run = a.has("dry-run");
    println!(
        "{} repository cleanup...",
        if dry_run { "Scanning for" } else { "Executing" }
    );
    let mut removed_dirs = 0;
    let mut removed_files = 0;

    let target_aix = ctx.repo.abs("target/aix");
    if target_aix.is_dir() {
        if dry_run {
            println!("  would remove: target/aix/");
        } else if std::fs::remove_dir_all(&target_aix).is_ok() {
            println!("  removed: target/aix/");
            removed_dirs += 1;
        }
    }

    let mut entries = Vec::new();
    walk_clean(&ctx.repo.root, "", &mut entries);
    for e in entries {
        if e.is_dir {
            if dry_run {
                println!("  would remove: {}/", e.rel);
            } else if std::fs::remove_dir_all(&e.path).is_ok() {
                println!("  removed: {}/", e.rel);
                removed_dirs += 1;
            }
        } else {
            if dry_run {
                println!("  would remove: {}", e.rel);
            } else if std::fs::remove_file(&e.path).is_ok() {
                println!("  removed: {}", e.rel);
                removed_files += 1;
            }
        }
    }

    if dry_run {
        println!("Dry run complete. Run `cargo aix clean` to remove these items.");
    } else {
        println!(
            "Cleanup complete: removed {} directories and {} files.",
            removed_dirs, removed_files
        );
    }
    Ok(true)
}

fn cmd_completions(_ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &[], &[])?;
    if a.has("help") || a.pos.is_empty() {
        print_cmd_help("completions");
        return Ok(true);
    }
    let shell = a.pos[0].to_lowercase();
    match shell.as_str() {
        "powershell" | "pwsh" => print!("{}", help::POWERSHELL_COMPLETIONS),
        "bash" => print!("{}", help::BASH_COMPLETIONS),
        "zsh" => print!("{}", help::ZSH_COMPLETIONS),
        "fish" => print!("{}", help::FISH_COMPLETIONS),
        other => {
            return Err(format!(
                "unsupported shell '{other}' (expected bash, zsh, fish, or powershell)"
            ));
        }
    }
    Ok(true)
}

/// verify_repo.py compatibility layer (same flags and default behaviour as the legacy script).
fn cmd_verify(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(
        argv,
        &["timeout", "workers", "path", "format", "max", "python"],
        &[
            "run",
            "check-symmetry",
            "check-links",
            "check-semantics",
            "parallel",
            "no-parallel",
            "check-candle",
            "check-kernel",
        ],
    )?;
    if a.has("help") {
        print_cmd_help("verify");
        return Ok(true);
    }
    let readme = ctx.readme()?;
    let explicit = a.has("check-symmetry")
        || a.has("check-links")
        || a.has("check-candle")
        || a.has("check-kernel");
    let mut only = Vec::new();
    if a.has("check-symmetry") {
        only.push("structure".to_string());
    }
    if a.has("check-links") {
        only.push("links".to_string());
    }
    if a.has("run") || a.has("check-semantics") || !explicit {
        only.push("notebooks".to_string());
    }
    let paths: Vec<String> = a.all("path").into_iter().filter(|p| p != ".").collect();
    let plan = CheckPlan {
        only,
        scope: ctx.scope(&paths, None, &readme)?,
        semantics: a.has("check-semantics"),
        fix: false,
    };
    let started = Instant::now();
    let mut ok = run_checks(
        ctx,
        &readme,
        &plan,
        Format::parse(a.get("format").unwrap_or("human"))?,
        a.num("max")?.unwrap_or(25),
        false,
    );
    if a.has("check-candle") {
        ok &= run::cargo_test(&ctx.repo.root, "ai-candle", None, None);
    }
    if a.has("check-kernel") {
        ok &= run::cargo_test(&ctx.repo.root, "ai-kernel", None, None);
    }
    if a.has("run") {
        let jobs = if a.has("no-parallel") {
            1
        } else {
            a.num("workers")?.unwrap_or_else(default_jobs)
        };
        let opts = run::NbOptions {
            scope: plan.scope.clone(),
            jobs,
            budget: a
                .num("timeout")?
                .unwrap_or(readme.contract.notebook.budget_seconds),
            grace: 45.0,
            python: a.get("python").map(str::to_string),
            verbose: false,
        };
        ok &= run::run_notebooks(&ctx.repo, &opts);
    }
    println!(
        "\n{} Repository validation {} in {:.2}s.",
        if ok { "[PASS]" } else { "[FAILED]" },
        if ok { "succeeded" } else { "failed" },
        started.elapsed().as_secs_f64()
    );
    Ok(ok)
}

fn cmd_ci(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &["format", "jobs", "python"], &[])?;
    if a.has("help") {
        print_cmd_help("ci");
        return Ok(true);
    }
    let readme = ctx.readme()?;
    let plan = CheckPlan {
        only: checks::ALL.iter().map(|s| s.to_string()).collect(),
        scope: Scope::default(),
        semantics: true,
        fix: false,
    };
    let mut ok = run_checks(
        ctx,
        &readme,
        &plan,
        Format::parse(a.get("format").unwrap_or("human"))?,
        50,
        false,
    );
    let opts = nb_options(ctx, &a, &readme, Scope::default())?;
    ok &= run::run_notebooks(&ctx.repo, &opts);
    for c in ["ai-architect-schema", "aix", "ai-candle", "ai-kernel"] {
        ok &= run::cargo_test(&ctx.repo.root, c, None, None);
    }
    Ok(ok)
}

fn cmd_doctor(ctx: &Ctx, argv: &[String]) -> Result<bool, String> {
    let a = Args::parse(argv, &["python"], &[])?;
    if a.has("help") {
        print_cmd_help("doctor");
        return Ok(true);
    }
    let readme = ctx.readme()?;
    println!("=== aix doctor: repository & environment diagnostic ===\n");
    println!("Repository root: {}", ctx.repo.root.display());
    println!(
        "Target topics:   {}",
        readme.contract.target_topics.unwrap_or(readme.topics.len())
    );
    println!("Curriculum:      {} topics listed", readme.topics.len());

    let (mut done, mut wip, mut planned) = (0, 0, 0);
    for t in &readme.topics {
        match sync::derived(&ctx.repo, &t.id).0 {
            curriculum::Status::Done => done += 1,
            curriculum::Status::Wip => wip += 1,
            curriculum::Status::Planned => planned += 1,
        }
    }
    println!(
        "Progress:        {} done, {} wip, {} planned ({:.1}%)\n",
        done,
        wip,
        planned,
        100.0 * done as f64 / readme.topics.len().max(1) as f64
    );

    println!("Rust toolchain:");
    let cargo_bin = run::cargo();
    match std::process::Command::new(&cargo_bin)
        .arg("--version")
        .output()
    {
        Ok(out) if out.status.success() => {
            let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
            println!("  [OK] cargo: {ver}");
        }
        _ => println!("  [WARN] cargo not found or failed"),
    }

    println!("Workspace crates:");
    for c in ["ai-architect-schema", "ai-candle", "ai-kernel", "aix"] {
        let p = format!("crates/{c}/Cargo.toml");
        if ctx.repo.exists(&p) {
            println!("  [OK] {c}");
        } else {
            println!("  [FAIL] {c} ({p} missing)");
        }
    }

    let py = run::python(&ctx.repo, a.get("python"));
    println!("\nPython environment:");
    println!("  Interpreter: {py}");
    match std::process::Command::new(&py)
        .args(["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"])
        .output()
    {
        Ok(out) if out.status.success() => {
            let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
            println!("  [OK] version: {ver}");
            for pkg in ["torch", "safetensors", "jax", "matplotlib", "numpy"] {
                let code = format!("import {pkg}; print({pkg}.__version__)");
                match std::process::Command::new(&py).args(["-c", &code]).output() {
                    Ok(o) if o.status.success() => {
                        let pver = String::from_utf8_lossy(&o.stdout).trim().to_string();
                        println!("  [OK] {pkg}: {pver}");
                    }
                    _ => println!("  [INFO] {pkg}: not installed (needed for ML training/interactive runs)"),
                }
            }
        }
        _ => println!("  [WARN] could not run '{py}' (needed for running notebooks and exporting weights)"),
    }

    println!("\nStatic check status:");
    let plan = CheckPlan {
        only: checks::ALL.iter().map(|s| s.to_string()).collect(),
        scope: Scope::default(),
        semantics: true,
        fix: false,
    };
    let ok = run_checks(ctx, &readme, &plan, Format::Human, 10, false);
    println!(
        "\nDoctor result: {}",
        if ok {
            "[PASS] repository healthy"
        } else {
            "[FAIL] repository issues detected (see diagnostics above)"
        }
    );
    Ok(ok)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first().cloned() else {
        print!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest = &argv[1..];
    if matches!(cmd.as_str(), "help" | "--help" | "-h") {
        if let Some(sub) = rest.first() {
            print_cmd_help(sub.as_str());
        } else {
            print!("{USAGE}");
        }
        return ExitCode::SUCCESS;
    }
    let root = match Repo::find_root(Path::new(".")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let mut ctx = Ctx {
        repo: Repo::load(root),
    };
    let result = match cmd.as_str() {
        "check" => cmd_check(&ctx, rest),
        "sync" => cmd_sync(&ctx, rest),
        "status" => cmd_status(&ctx, rest),
        "next" => cmd_next(&ctx, rest),
        "run-notebooks" => cmd_run_notebooks(&ctx, rest),
        "test" => cmd_test(&ctx, rest),
        "scaffold" => cmd_scaffold(&mut ctx, rest),
        "notebook" => cmd_notebook(&ctx, rest),
        "weights" => cmd_weights(&ctx, rest),
        "verify" => cmd_verify(&ctx, rest),
        "ci" => cmd_ci(&ctx, rest),
        "doctor" => cmd_doctor(&ctx, rest),
        "clean" => cmd_clean(&ctx, rest),
        "completions" => cmd_completions(&ctx, rest),
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
