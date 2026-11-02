//! Process orchestration: notebook execution (one isolated Python process per notebook with a
//! hard wall-clock kill) and cargo test invocations.

use std::collections::{BTreeMap, VecDeque};
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::repo::{Repo, Scope};

pub const WORKER: &str = "scripts/aix_nb_worker.py";

pub struct NbOptions {
    pub scope: Scope,
    pub jobs: usize,
    pub budget: f64,
    /// Extra wall-clock allowance for interpreter start-up and imports before a hard kill.
    pub grace: f64,
    pub python: Option<String>,
    pub verbose: bool,
}

struct NbResult {
    rel: String,
    ok: bool,
    elapsed: f64,
    error: String,
    output: String,
}

/// Python interpreter: --python, $AIX_PYTHON, the repo .venv, then `python` on PATH.
pub fn python(repo: &Repo, explicit: Option<&str>) -> String {
    if let Some(p) = explicit {
        return p.to_string();
    }
    if let Ok(p) = std::env::var("AIX_PYTHON") {
        if !p.is_empty() {
            return p;
        }
    }
    for cand in [
        ".venv/Scripts/python.exe",
        ".venv/bin/python",
        "venv/Scripts/python.exe",
        "venv/bin/python",
    ] {
        let p = repo.abs(cand);
        if p.is_file() {
            return p.to_string_lossy().to_string();
        }
    }
    #[cfg(windows)]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let base = std::path::PathBuf::from(local)
                .join("Programs")
                .join("Python");
            if base.is_dir() {
                if let Ok(entries) = std::fs::read_dir(base) {
                    let mut pythons: Vec<std::path::PathBuf> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path().join("python.exe"))
                        .filter(|p| p.is_file())
                        .collect();
                    pythons.sort();
                    if let Some(best) = pythons.pop() {
                        return best.to_string_lossy().to_string();
                    }
                }
            }
        }
    }
    "python".to_string()
}

fn drain(mut r: impl Read + Send + 'static) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = r.read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).to_string()
    })
}

/// Waits for a child with a deadline; returns (exit success, timed_out).
fn wait_deadline(child: &mut Child, limit: Duration) -> (bool, bool) {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return (status.success(), false),
            Ok(None) if start.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                return (false, true);
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(_) => return (false, false),
        }
    }
}

fn run_one(repo: &Repo, py: &str, rel: &str, opts: &NbOptions) -> NbResult {
    let fail = |error: String, output: String| NbResult {
        rel: rel.to_string(),
        ok: false,
        elapsed: 0.0,
        error,
        output,
    };
    let spawned = Command::new(py)
        .arg(repo.abs(WORKER))
        .arg(repo.abs(rel))
        .current_dir(&repo.root)
        .env("MPLBACKEND", "Agg")
        .env("QT_QPA_PLATFORM", "offscreen")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => return fail(format!("failed to start '{py}': {e}"), String::new()),
    };
    let out_h = drain(child.stdout.take().expect("piped stdout"));
    let err_h = drain(child.stderr.take().expect("piped stderr"));
    let (_, timed_out) = wait_deadline(
        &mut child,
        Duration::from_secs_f64(opts.budget + opts.grace),
    );
    let stdout = out_h.join().unwrap_or_default();
    let stderr = err_h.join().unwrap_or_default();
    let output = format!("{stdout}{stderr}");
    if timed_out {
        return fail(
            format!(
                "killed after {:.1}s wall clock (budget {:.1}s + {:.0}s start-up grace)",
                opts.budget + opts.grace,
                opts.budget,
                opts.grace
            ),
            output,
        );
    }
    let result = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("AIX_RESULT "))
        .and_then(|j| serde_json::from_str::<serde_json::Value>(j).ok());
    let Some(r) = result else {
        return fail(
            "worker exited without a result (crash during start-up?)".into(),
            output,
        );
    };
    let elapsed = r.get("elapsed").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let mut ok = r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut error = r
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    if ok && elapsed > opts.budget {
        ok = false;
        error = format!("took {elapsed:.2}s (budget {:.1}s)", opts.budget);
    }
    NbResult {
        rel: rel.to_string(),
        ok,
        elapsed,
        error,
        output,
    }
}

/// Executes every notebook in scope; returns true when all pass within budget.
pub fn run_notebooks(repo: &Repo, opts: &NbOptions) -> bool {
    let list: Vec<String> = repo
        .files_with("notebooks", "ipynb")
        .filter(|f| opts.scope.includes(f))
        .cloned()
        .collect();
    if list.is_empty() {
        println!("No notebooks in scope.");
        return true;
    }
    if !repo.abs(WORKER).is_file() {
        println!("[FAIL] notebook worker {WORKER} is missing");
        return false;
    }
    let py = python(repo, opts.python.as_deref());
    let jobs = opts.jobs.max(1).min(list.len());
    println!(
        "Executing {} notebook(s) with {jobs} worker process(es), budget {:.1}s each ({py})",
        list.len(),
        opts.budget
    );

    let started = Instant::now();
    let queue = Arc::new(Mutex::new(list.into_iter().collect::<VecDeque<_>>()));
    let results: Arc<Mutex<Vec<NbResult>>> = Arc::new(Mutex::new(Vec::new()));
    thread::scope(|s| {
        for _ in 0..jobs {
            let (queue, results, py) = (Arc::clone(&queue), Arc::clone(&results), py.clone());
            s.spawn(move || loop {
                let Some(rel) = queue.lock().ok().and_then(|mut q| q.pop_front()) else {
                    break;
                };
                let r = run_one(repo, &py, &rel, opts);
                if r.ok {
                    println!("[PASS] {} ({:.2}s)", r.rel, r.elapsed);
                    if opts.verbose {
                        println!("{}", indent(&r.output));
                    }
                } else {
                    println!("[FAIL] {} - {}", r.rel, r.error);
                    println!("{}", indent(tail(&r.output, 25)));
                }
                if let Ok(mut v) = results.lock() {
                    v.push(r);
                }
            });
        }
    });

    let results = Arc::try_unwrap(results)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default();
    let failed = results.iter().filter(|r| !r.ok).count();
    let timings: BTreeMap<&str, f64> = results
        .iter()
        .map(|r| (r.rel.as_str(), (r.elapsed * 1000.0).round() / 1000.0))
        .collect();
    let out_path = repo.abs("target/aix/notebook_timings.json");
    if let Some(dir) = out_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(
        &out_path,
        serde_json::to_string_pretty(&timings).unwrap_or_default(),
    );

    let mut slow: Vec<&NbResult> = results.iter().collect();
    slow.sort_by(|a, b| b.elapsed.total_cmp(&a.elapsed));
    println!(
        "Slowest: {}",
        slow.iter()
            .take(5)
            .map(|r| format!("{} {:.2}s", crate::repo::file_name(&r.rel), r.elapsed))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "{} {}/{} notebooks passed in {:.1}s (timings: target/aix/notebook_timings.json)",
        if failed == 0 { "[PASS]" } else { "[FAIL]" },
        results.len() - failed,
        results.len(),
        started.elapsed().as_secs_f64()
    );
    failed == 0
}

fn tail(s: &str, n: usize) -> &str {
    let idx = s
        .trim_end()
        .rmatch_indices('\n')
        .nth(n.saturating_sub(1))
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    &s[idx..]
}

fn indent(s: &str) -> String {
    s.trim_end()
        .lines()
        .map(|l| format!("       | {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

/// `cargo test -p <crate> --tests [-j N] [-- filter]` with inherited stdio.
pub fn cargo_test(root: &Path, krate: &str, filter: Option<&str>, jobs: Option<usize>) -> bool {
    let mut cmd = Command::new(cargo());
    cmd.current_dir(root).args(["test", "-p", krate, "--tests"]);
    if let Some(j) = jobs {
        cmd.args(["-j", &j.to_string()]);
    }
    if let Some(f) = filter {
        cmd.args(["--", f]);
    }
    println!(
        "$ cargo test -p {krate} --tests{}",
        filter.map(|f| format!(" -- {f}")).unwrap_or_default()
    );
    match cmd.status() {
        Ok(s) if s.success() => {
            println!("[PASS] {krate} tests");
            true
        }
        Ok(s) => {
            println!("[FAIL] {krate} tests (exit {})", s.code().unwrap_or(-1));
            false
        }
        Err(e) => {
            println!("[FAIL] could not run cargo: {e}");
            false
        }
    }
}

/// Runs `python <script> [args]` from the repo root with inherited stdio.
pub fn python_script(repo: &Repo, py: &str, script: &str, args: &[&str]) -> bool {
    println!("$ python {script} {}", args.join(" "));
    match Command::new(py)
        .arg(repo.abs(script))
        .args(args)
        .current_dir(&repo.root)
        .env("MPLBACKEND", "Agg")
        .status()
    {
        Ok(s) if s.success() => true,
        Ok(_) => false,
        Err(e) => {
            println!("[FAIL] could not start '{py}': {e}");
            false
        }
    }
}
