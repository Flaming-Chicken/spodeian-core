//! # Spodeian Provenance
//!
//! Whole-token license verification, unified `LedgerRecord` schema,
//! and deterministic `ATTRIBUTIONS.txt` generation for audio/data pipelines.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// Errors raised while validating provenance records or license tokens.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    #[error("missing required field `{0}` on record `{1}`")]
    MissingField(&'static str, String),
    #[error("disallowed or unrecognized license `{license}` on record `{id}`")]
    DisallowedLicense { id: String, license: String },
    #[error("duplicate record id `{0}`")]
    DuplicateId(String),
    #[error("duplicate sha256 digest `{0}`")]
    DuplicateSha256(String),
    #[error("contributor grant `{0}` missing explicit rights_warranty")]
    MissingRightsWarranty(String),
}

/// Origin category for a curated or synthetic asset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceOrigin {
    Freesound,
    Synthetic,
    ContributorGrant,
    PublicDomainArchive,
    ExternalVerified,
}

/// Unified provenance ledger record folding download, source, metadata, and attribution fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerRecord {
    pub id: String,
    pub title: String,
    pub author: String,
    pub origin: SourceOrigin,
    pub license_spdx: String,
    pub source_url: Option<String>,
    pub sha256: String,
    pub duration_secs: f64,
    pub sample_rate_hz: u32,
    pub surface: Option<String>,
    pub rights_warranty: bool,
    pub ingested_at_unix: u64,
}

/// Policy-enforcing whole-token license verifier.
#[derive(Debug, Clone)]
pub struct LicenseVerifier {
    allowed_exact: HashSet<String>,
    forbidden_tokens: HashSet<String>,
}

impl Default for LicenseVerifier {
    fn default() -> Self {
        Self::commercial_safe_default()
    }
}

impl LicenseVerifier {
    /// Standard D2/D4-compliant allowlist (CC0, Public Domain, CC-BY 3.0/4.0, Synthetic, Spodeian-D4)
    /// with whole-token rejection of `NC` (NonCommercial) and `ND` (NoDerivatives) clauses.
    pub fn commercial_safe_default() -> Self {
        let allowed_exact = [
            "CC0-1.0",
            "CC0",
            "PD",
            "PUBLIC-DOMAIN",
            "CC-BY-3.0",
            "CC-BY-4.0",
            "SYNTHETIC",
            "SPODEIAN-D4",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();

        let forbidden_tokens = ["NC", "ND", "SA", "PROPRIETARY", "ALL-RIGHTS-RESERVED"]
            .into_iter()
            .map(str::to_string)
            .collect();

        Self {
            allowed_exact,
            forbidden_tokens,
        }
    }

    /// Normalizes a license label into uppercase hyphen-separated canonical form.
    pub fn normalize(raw: &str) -> String {
        raw.trim()
            .to_uppercase()
            .replace([' ', '_'], "-")
    }

    /// Checks whether `license` is permitted using whole-token matching (never substring matching).
    pub fn is_allowed(&self, license: &str) -> bool {
        let normalized = Self::normalize(license);
        if normalized.is_empty() {
            return false;
        }
        let tokens: Vec<&str> = normalized
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '.')
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.iter().any(|t| self.forbidden_tokens.contains(*t)) {
            return false;
        }
        self.allowed_exact.contains(&normalized)
    }

    /// Returns `true` if the license requires human attribution (e.g., CC-BY-3.0 / CC-BY-4.0).
    pub fn requires_attribution(&self, license: &str) -> bool {
        let norm = Self::normalize(license);
        norm == "CC-BY-3.0" || norm == "CC-BY-4.0"
    }
}

/// Validates a full ledger for required metadata, duplicate IDs/hashes, D4 contributor warranties,
/// and whole-token license compliance.
pub fn validate_ledger(
    records: &[LedgerRecord],
    verifier: &LicenseVerifier,
) -> Result<(), ProvenanceError> {
    let mut seen_ids = HashSet::with_capacity(records.len());
    let mut seen_hashes = HashSet::with_capacity(records.len());

    for rec in records {
        if rec.id.trim().is_empty() {
            return Err(ProvenanceError::MissingField("id", rec.title.clone()));
        }
        if rec.title.trim().is_empty() {
            return Err(ProvenanceError::MissingField("title", rec.id.clone()));
        }
        if rec.sha256.trim().is_empty() {
            return Err(ProvenanceError::MissingField("sha256", rec.id.clone()));
        }
        if !seen_ids.insert(rec.id.clone()) {
            return Err(ProvenanceError::DuplicateId(rec.id.clone()));
        }
        if !seen_hashes.insert(rec.sha256.clone()) {
            return Err(ProvenanceError::DuplicateSha256(rec.sha256.clone()));
        }
        if rec.origin == SourceOrigin::ContributorGrant && !rec.rights_warranty {
            return Err(ProvenanceError::MissingRightsWarranty(rec.id.clone()));
        }
        if !verifier.is_allowed(&rec.license_spdx) {
            return Err(ProvenanceError::DisallowedLicense {
                id: rec.id.clone(),
                license: rec.license_spdx.clone(),
            });
        }
    }
    Ok(())
}

/// Renders deterministic `ATTRIBUTIONS.txt` sorted by `id` for all records requiring attribution
/// (or all records when `include_cc0` is `true`).
pub fn render_attributions_txt(
    records: &[LedgerRecord],
    verifier: &LicenseVerifier,
    include_cc0: bool,
) -> String {
    let mut sorted: Vec<&LedgerRecord> = records
        .iter()
        .filter(|r| include_cc0 || verifier.requires_attribution(&r.license_spdx))
        .collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));

    let mut out = String::from("# Asset Attributions\n\n");
    if sorted.is_empty() {
        out.push_str("No attribution-required assets in this release.\n");
        return out;
    }
    for rec in sorted {
        let url = rec.source_url.as_deref().unwrap_or("local/internal");
        out.push_str(&format!(
            "- [{}] \"{}\" by {} — {} ({})\n",
            rec.id, rec.title, rec.author, rec.license_spdx, url
        ));
    }
    out
}
