//! Parses the shared planning document (README.md): the topic curriculum checklist and the
//! machine-readable contract block. README.md is the single source of truth for both.

use serde::Deserialize;

use crate::repo::{Repo, is_numbered, parent, track_ext};

pub const README: &str = "README.md";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    Planned,
    Wip,
    Done,
}

impl Status {
    pub fn mark(self) -> char {
        match self {
            Status::Planned => ' ',
            Status::Wip => '~',
            Status::Done => 'x',
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Status::Planned => "planned",
            Status::Wip => "wip",
            Status::Done => "done",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Topic {
    /// `<category>[/<subfolder>]/<NN_slug>` (path inside every track, without extension).
    pub id: String,
    pub title: String,
    pub status: Status,
    /// 0-based README line index.
    pub line: usize,
}

impl Topic {
    pub fn category(&self) -> &str {
        self.id.split('/').next().unwrap_or("")
    }

    pub fn subfolder(&self) -> Option<&str> {
        let parts: Vec<&str> = self.id.split('/').collect();
        (parts.len() == 3).then(|| parts[1])
    }

    pub fn slug(&self) -> &str {
        self.id.rsplit('/').next().unwrap_or("")
    }

    pub fn num(&self) -> &str {
        &self.slug()[..2]
    }

    /// Ordering slot (`parent/NN`); two topics must never share one.
    pub fn slot(&self) -> String {
        format!("{}/{}", parent(&self.id), self.num())
    }

    pub fn path(&self, track: &str) -> String {
        format!("{track}/{}.{}", self.id, track_ext(track).unwrap_or(""))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contract {
    /// Total number of topics the finished catalog should contain (for progress reporting).
    #[serde(default)]
    pub target_topics: Option<usize>,
    pub notebook: NotebookContract,
    pub markdown: MarkdownContract,
    pub forbidden_calls: Vec<String>,
    pub weights: WeightsContract,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotebookContract {
    /// `[cell_type, description]` for each required cell, in order.
    pub cells: Vec<(String, String)>,
    pub max_lines: usize,
    pub budget_seconds: f64,
    pub semantics: Vec<SemanticRule>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticRule {
    /// 1-based cell index.
    pub cell: usize,
    pub any: Vec<String>,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkdownContract {
    /// When false, missing sections are reported as warnings only.
    pub enforce: bool,
    /// Required `##` sections; `A|B` accepts either keyword (case-insensitive substring).
    pub sections: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightsContract {
    /// Each inner list is a set of alternative tensor names, one of which must be present.
    pub required_tensors: Vec<Vec<String>>,
}

pub struct Readme {
    pub raw: String,
    pub topics: Vec<Topic>,
    pub contract: Contract,
    /// Problems found while parsing (reported by `aix check`).
    pub problems: Vec<(usize, String)>,
}

impl Readme {
    pub fn topic(&self, id: &str) -> Option<&Topic> {
        self.topics.iter().find(|t| t.id == id)
    }

    /// Resolves a user-supplied topic reference: full id, slug, or unique suffix.
    pub fn resolve(&self, query: &str) -> Result<&Topic, String> {
        let q = query
            .trim_end_matches(".ipynb")
            .trim_start_matches("notebooks/")
            .trim_matches('/')
            .replace('\\', "/");
        if let Some(t) = self.topic(&q) {
            return Ok(t);
        }
        let hits: Vec<&Topic> = self
            .topics
            .iter()
            .filter(|t| {
                t.id.ends_with(&format!("/{q}"))
                    || t.slug() == q
                    || crate::repo::strip_num(t.slug()) == q
            })
            .collect();
        match hits.as_slice() {
            [t] => Ok(t),
            [] => Err(format!("topic '{query}' is not in the README curriculum")),
            many => Err(format!(
                "topic '{query}' is ambiguous: {}",
                many.iter()
                    .map(|t| t.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

pub fn lines(raw: &str) -> Vec<&str> {
    raw.split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect()
}

/// Indices of the `<!-- aix:NAME:begin -->` / `<!-- aix:NAME:end -->` marker lines.
pub fn block(lines: &[&str], name: &str) -> Option<(usize, usize)> {
    let begin = format!("<!-- aix:{name}:begin -->");
    let end = format!("<!-- aix:{name}:end -->");
    let b = lines.iter().position(|l| l.trim() == begin)?;
    let e = lines[b..].iter().position(|l| l.trim() == end)? + b;
    Some((b, e))
}

/// Parses `- [x] **NN. Title** — [`slug`](notebooks/<id>.ipynb) ...` checklist entries.
pub fn parse_entry(line: &str) -> Option<(Status, String, String)> {
    let rest = line.trim_start().strip_prefix("- [")?;
    let mut chars = rest.chars();
    let status = match chars.next()? {
        ' ' => Status::Planned,
        '~' => Status::Wip,
        'x' | 'X' => Status::Done,
        _ => return None,
    };
    let rest = chars.as_str().strip_prefix("] **")?;
    let end = rest.find("**")?;
    let raw_title = &rest[..end];
    let title = match raw_title.split_once(". ") {
        Some((n, t)) if n.chars().all(|c| c.is_ascii_digit()) => t,
        _ => raw_title,
    };
    let marker = "](notebooks/";
    let start = rest.find(marker)? + marker.len();
    let tail = &rest[start..];
    let id = &tail[..tail.find(".ipynb)")?];
    Some((status, title.trim().to_string(), id.to_string()))
}

pub fn load(repo: &Repo) -> Result<Readme, String> {
    let raw = repo.read(README)?;
    let ls = lines(&raw);
    let mut problems = Vec::new();

    let (cb, ce) = block(&ls, "curriculum").ok_or(
        "README.md has no `<!-- aix:curriculum:begin -->` ... `<!-- aix:curriculum:end -->` block",
    )?;
    let mut topics = Vec::new();
    for (i, line) in ls.iter().enumerate().take(ce).skip(cb + 1) {
        let t = line.trim_start();
        if !t.starts_with("- [") {
            continue;
        }
        match parse_entry(line) {
            Some((status, title, id)) => {
                let segs: Vec<&str> = id.split('/').collect();
                if !(2..=3).contains(&segs.len()) || !segs.iter().all(|s| is_numbered(s)) {
                    problems.push((i, format!("topic id '{id}' must be <NN_category>[/<NN_subfolder>]/<NN_slug>")));
                    continue;
                }
                topics.push(Topic { id, title, status, line: i });
            }
            None => problems.push((
                i,
                "unparseable checklist entry; expected `- [ ] **NN. Title** — [`NN_slug`](notebooks/<id>.ipynb)`".into(),
            )),
        }
    }

    let (kb, ke) = block(&ls, "contract").ok_or(
        "README.md has no `<!-- aix:contract:begin -->` ... `<!-- aix:contract:end -->` block",
    )?;
    let body = &ls[kb + 1..ke];
    let fence_start = body
        .iter()
        .position(|l| l.trim_start().starts_with("```"))
        .ok_or("contract block has no ```json fence")?;
    let fence_len = body[fence_start + 1..]
        .iter()
        .position(|l| l.trim_start().starts_with("```"))
        .ok_or("contract block fence is not closed")?;
    let json = body[fence_start + 1..fence_start + 1 + fence_len].join("\n");
    let contract: Contract =
        serde_json::from_str(&json).map_err(|e| format!("README contract JSON is invalid: {e}"))?;

    Ok(Readme {
        raw,
        topics,
        contract,
        problems,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_checklist_entries() {
        let (s, title, id) = parse_entry(
            "- [x] **01. RNN, LSTM & GRU** — [`01_rnn`](notebooks/05_architectures/03_recurrent/01_rnn.ipynb)",
        )
        .unwrap();
        assert_eq!(s, Status::Done);
        assert_eq!(title, "RNN, LSTM & GRU");
        assert_eq!(id, "05_architectures/03_recurrent/01_rnn");
        assert_eq!(
            parse_entry("- [~] **02. X** — [`02_x`](notebooks/00_f/02_x.ipynb)")
                .unwrap()
                .0,
            Status::Wip
        );
        assert!(parse_entry("- plain bullet").is_none());
    }
}
