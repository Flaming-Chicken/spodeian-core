//! Diagnostics and output formatting (human, JSON, GitHub Actions annotations).

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Warning,
    Error,
}

#[derive(Clone, Debug)]
pub struct Diag {
    pub level: Level,
    pub check: &'static str,
    pub file: String,
    pub line: Option<usize>,
    pub msg: String,
    pub fix: Option<String>,
}

impl Diag {
    pub fn error(check: &'static str, file: impl Into<String>, msg: impl Into<String>) -> Self {
        Self {
            level: Level::Error,
            check,
            file: file.into(),
            line: None,
            msg: msg.into(),
            fix: None,
        }
    }

    pub fn warn(check: &'static str, file: impl Into<String>, msg: impl Into<String>) -> Self {
        Self {
            level: Level::Warning,
            check,
            file: file.into(),
            line: None,
            msg: msg.into(),
            fix: None,
        }
    }

    pub fn at(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }

    fn location(&self) -> String {
        match self.line {
            Some(l) => format!("{}:{}", self.file, l),
            None => self.file.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Human,
    Json,
    Github,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "human" => Ok(Self::Human),
            "json" => Ok(Self::Json),
            "github" => Ok(Self::Github),
            other => Err(format!(
                "unknown --format '{other}' (expected human|json|github)"
            )),
        }
    }
}

/// Result of one named check section.
pub struct Section {
    pub name: &'static str,
    pub diags: Vec<Diag>,
    pub seconds: f64,
}

impl Section {
    pub fn errors(&self) -> usize {
        self.diags
            .iter()
            .filter(|d| d.level == Level::Error)
            .count()
    }

    pub fn warnings(&self) -> usize {
        self.diags
            .iter()
            .filter(|d| d.level == Level::Warning)
            .count()
    }
}

/// Emits all sections; returns true when no errors were found.
pub fn emit(sections: &[Section], format: Format, max_per_check: usize, quiet: bool) -> bool {
    let ok = sections.iter().all(|s| s.errors() == 0);
    match format {
        Format::Json => {
            let items: Vec<serde_json::Value> = sections
                .iter()
                .flat_map(|s| s.diags.iter())
                .map(|d| {
                    serde_json::json!({
                        "level": if d.level == Level::Error { "error" } else { "warning" },
                        "check": d.check,
                        "file": d.file,
                        "line": d.line,
                        "message": d.msg,
                        "fix": d.fix,
                    })
                })
                .collect();
            let summary: BTreeMap<&str, serde_json::Value> = sections
                .iter()
                .map(|s| {
                    (s.name, serde_json::json!({"errors": s.errors(), "warnings": s.warnings(), "seconds": s.seconds}))
                })
                .collect();
            let out = serde_json::json!({"ok": ok, "summary": summary, "diagnostics": items});
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        }
        Format::Github | Format::Human => {
            if format == Format::Github {
                for d in sections.iter().flat_map(|s| s.diags.iter()) {
                    let kind = if d.level == Level::Error {
                        "error"
                    } else {
                        "warning"
                    };
                    let line = d.line.map(|l| format!(",line={l}")).unwrap_or_default();
                    let msg = d.msg.replace('%', "%25").replace('\n', "%0A");
                    println!(
                        "::{kind} file={}{line},title=aix {}::{msg}",
                        d.file, d.check
                    );
                }
            }
            for s in sections {
                let (e, w) = (s.errors(), s.warnings());
                let status = if e > 0 {
                    "FAIL"
                } else if w > 0 {
                    "WARN"
                } else {
                    "PASS"
                };
                if quiet && status == "PASS" {
                    continue;
                }
                println!(
                    "[{status}] {:<10} {e} error(s), {w} warning(s) ({:.2}s)",
                    s.name, s.seconds
                );
                let mut sorted: Vec<&Diag> = s.diags.iter().collect();
                sorted.sort_by(|a, b| {
                    b.level
                        .cmp(&a.level)
                        .then(a.file.cmp(&b.file))
                        .then(a.line.cmp(&b.line))
                });
                for (shown, d) in sorted.iter().enumerate() {
                    if shown >= max_per_check {
                        println!(
                            "       ... {} more (use --max N to show more)",
                            s.diags.len() - shown
                        );
                        break;
                    }
                    let tag = if d.level == Level::Error {
                        "error"
                    } else {
                        "warn "
                    };
                    match &d.fix {
                        Some(f) => println!("       {tag} {}: {} (fix: {f})", d.location(), d.msg),
                        None => println!("       {tag} {}: {}", d.location(), d.msg),
                    }
                }
            }
        }
    }
    ok
}
