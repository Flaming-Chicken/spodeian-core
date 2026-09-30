//! Dataset metadata lineage distillation engine for academic, ethical, and software citations.
//!
//! Converts raw `DatasetMetadata` records into standard schemas:
//! - BibTeX (`@misc{...}` / `@dataset{...}`)
//! - Citation File Format (CFF 1.2.0)
//! - Plain text (APA / Chicago style)
//! - Schema.org Dataset JSON-LD

use crate::registry::DatasetMetadata;
use serde::{Deserialize, Serialize};

/// Citation format targets.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CitationFormat {
    BibTeX,
    Cff,
    Apa,
    JsonLd,
}

/// Metadata Distillation Engine for Dataset Attribution and Lineage.
#[doc = "**Stability: Experimental** — API subject to change"]
pub struct CitationDistiller;

impl CitationDistiller {
    /// Distills a single dataset partition metadata record into the requested citation format.
    pub fn distill(meta: &DatasetMetadata, format: CitationFormat) -> String {
        match format {
            CitationFormat::BibTeX => Self::to_bibtex(meta),
            CitationFormat::Cff => Self::to_cff(meta),
            CitationFormat::Apa => Self::to_apa(meta),
            CitationFormat::JsonLd => Self::to_json_ld(meta),
        }
    }

    /// Distills a collection of dataset partition metadata records.
    pub fn distill_manifest(records: &[DatasetMetadata], format: CitationFormat) -> String {
        match format {
            CitationFormat::BibTeX => {
                records.iter().map(Self::to_bibtex).collect::<Vec<_>>().join("\n\n")
            }
            CitationFormat::Cff => {
                let mut out = String::from("cff-version: 1.2.0\nmessage: \"If you use these datasets, please cite them as below.\"\nreferences:\n");
                for meta in records {
                    let indented = Self::to_cff(meta)
                        .lines()
                        .map(|l| format!("  {l}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    out.push_str(&format!("- type: data\n{indented}\n"));
                }
                out
            }
            CitationFormat::Apa => {
                records.iter().map(Self::to_apa).collect::<Vec<_>>().join("\n")
            }
            CitationFormat::JsonLd => {
                let items: Vec<serde_json::Value> = records
                    .iter()
                    .filter_map(|m| serde_json::from_str(&Self::to_json_ld(m)).ok())
                    .collect();
                let graph = serde_json::json!({
                    "@context": "https://schema.org",
                    "@graph": items
                });
                serde_json::to_string_pretty(&graph).unwrap_or_default()
            }
        }
    }

    /// Generates standard BibTeX `@misc` / `@dataset` entry.
    pub fn to_bibtex(meta: &DatasetMetadata) -> String {
        let key = format!("{}_{}", sanitize_key(&meta.dataset_id), sanitize_key(&meta.partition_key));
        let title = format!("Dataset Partition: {} ({})", meta.dataset_id, meta.partition_key);
        let author = meta.author.as_deref().unwrap_or("Spodeian Ecosystem Contributor");
        let year = epoch_to_year(meta.created_at_epoch_s);
        let note = format!("SHA256: {}; Size: {} bytes", meta.sha256_checksum, meta.byte_size);

        let mut lines = Vec::new();
        lines.push(format!("@misc{{{key},"));
        lines.push(format!("  title = {{{{{title}}}}},"));
        lines.push(format!("  author = {{{author}}},"));
        lines.push(format!("  year = {{{year}}},"));
        if let Some(doi) = &meta.doi_or_url {
            lines.push(format!("  howpublished = {{\\url{{{doi}}}}},"));
        }
        if let Some(lic) = &meta.license {
            lines.push(format!("  license = {{{lic}}},"));
        }
        lines.push(format!("  note = {{{note}}}"));
        lines.push("}".to_string());

        lines.join("\n")
    }

    /// Generates Citation File Format (CFF) representation.
    pub fn to_cff(meta: &DatasetMetadata) -> String {
        let title = format!("Dataset: {} - Partition: {}", meta.dataset_id, meta.partition_key);
        let author = meta.author.as_deref().unwrap_or("Spodeian Ecosystem Contributor");
        let year = epoch_to_year(meta.created_at_epoch_s);
        let license = meta.license.as_deref().unwrap_or("proprietary");

        let mut lines = Vec::new();
        lines.push(format!("title: \"{title}\""));
        lines.push("authors:".to_string());
        lines.push(format!("  - name: \"{author}\""));
        lines.push(format!("year: {year}"));
        lines.push(format!("license: \"{license}\""));
        if let Some(url) = &meta.doi_or_url {
            lines.push(format!("url: \"{url}\""));
        }
        lines.push("keywords:".to_string());
        lines.push("  - \"machine-learning\"".to_string());
        lines.push("  - \"data-lake\"".to_string());
        lines.push(format!("  - \"sha256:{}\"", meta.sha256_checksum));

        lines.join("\n")
    }

    /// Generates plain-text APA style reference string.
    pub fn to_apa(meta: &DatasetMetadata) -> String {
        let author = meta.author.as_deref().unwrap_or("Spodeian Ecosystem Contributor");
        let year = epoch_to_year(meta.created_at_epoch_s);
        let title = format!("{}: {}", meta.dataset_id, meta.partition_key);
        let url_part = if let Some(url) = &meta.doi_or_url {
            format!(" {url}")
        } else {
            String::new()
        };
        let hash_part = format!(" [Checksum: sha256:{}]", &meta.sha256_checksum[..12.min(meta.sha256_checksum.len())]);

        format!("{author} ({year}). {title} [Data set]. Spodeian Data Lake.{hash_part}{url_part}")
    }

    /// Generates Schema.org JSON-LD Dataset entry.
    pub fn to_json_ld(meta: &DatasetMetadata) -> String {
        let author = meta.author.as_deref().unwrap_or("Spodeian Ecosystem Contributor");
        let license = meta.license.as_deref().unwrap_or("https://choosealicense.com/no-permission/");
        let url = meta.doi_or_url.as_deref().unwrap_or("urn:spodeian:lake");

        let json_obj = serde_json::json!({
            "@context": "https://schema.org",
            "@type": "Dataset",
            "name": format!("{}: {}", meta.dataset_id, meta.partition_key),
            "description": format!("Partition of dataset '{}' with {} records ({} bytes).", meta.dataset_id, meta.record_count, meta.byte_size),
            "creator": {
                "@type": "Person",
                "name": author
            },
            "license": license,
            "url": url,
            "identifier": format!("sha256:{}", meta.sha256_checksum),
            "distribution": [
                {
                    "@type": "DataDownload",
                    "encodingFormat": "application/vnd.apache.parquet",
                    "contentSize": format!("{} B", meta.byte_size)
                }
            ]
        });

        serde_json::to_string_pretty(&json_obj).unwrap_or_default()
    }
}

fn sanitize_key(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

fn epoch_to_year(epoch_s: u64) -> u32 {
    // 1970 + epoch_s / (365.25 * 86400)
    1970 + (epoch_s as f64 / 31_557_600.0) as u32
}
