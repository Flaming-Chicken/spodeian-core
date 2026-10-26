//! Partition registry, write-ahead logging (WAL), and provenance metadata tracking for Spodeian Data Lake.
//!
//! **Stability: Experimental** — API subject to change

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Dataset not found: {0}:{1}")]
    NotFound(String, String),
}

/// Metadata record for a single dataset partition in the Spodeian Data Lake.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetMetadata {
    pub dataset_id: String,
    pub partition_key: String,
    pub file_path: PathBuf,
    pub record_count: usize,
    pub byte_size: u64,
    pub sha256_checksum: String,
    pub created_at_epoch_s: u64,
    pub last_accessed_epoch_s: u64,
    pub ttl_seconds: Option<u64>,

    // Provenance, attribution & ethical metadata lineage
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub doi_or_url: Option<String>,
    #[serde(default)]
    pub license: Option<String>,

    // Value-aware training telemetry & regime retention flags
    pub training_utility: f64, // Reconstruction/loss signal in [0.0, 1.0] (high = model still learning)
    pub anomaly_score: f64,    // Tail-risk / rare regime index in [0.0, 1.0] (high = crisis/rare)
    pub training_epochs_seen: usize, // Saturation count (higher = diminishing returns)
    pub is_pinned: bool, // If true, permanently exempt from automated eviction (e.g. crisis memories)
}

impl DatasetMetadata {
    /// Determines whether the partition has exceeded its absolute time-to-live.
    pub fn is_stale(&self, current_epoch_s: u64) -> bool {
        // Pinned partitions or partitions with significant training utility are never stale
        if self.is_pinned || self.training_utility > 0.4 || self.anomaly_score > 0.6 {
            return false;
        }
        if let Some(ttl) = self.ttl_seconds {
            current_epoch_s > self.created_at_epoch_s.saturating_add(ttl)
        } else {
            false
        }
    }

    /// Computes the Composite Retention Priority score P.
    /// Higher score = more valuable to retain for ongoing model training.
    /// Lowest scores are selected first for pruning/eviction.
    pub fn compute_retention_priority(&self, current_epoch_s: u64) -> f64 {
        if self.is_pinned {
            return f64::MAX;
        }

        let utility_weight = 0.45;
        let anomaly_weight = 0.35;
        let recency_weight = 0.15;
        let saturation_penalty = 0.05;

        // Recency factor decays logarithmically over time
        let age_seconds = current_epoch_s.saturating_sub(self.last_accessed_epoch_s) as f64;
        let recency_factor = 1.0 / (1.0 + (age_seconds / 86400.0).ln_1p());

        // Saturation factor increases as the model sees this partition repeatedly without high error
        let saturation = (self.training_epochs_seen as f64 / 20.0).min(1.0);

        let priority = (utility_weight * self.training_utility.clamp(0.0, 1.0))
            + (anomaly_weight * self.anomaly_score.clamp(0.0, 1.0))
            + (recency_weight * recency_factor)
            - (saturation_penalty * saturation);

        priority.max(0.0)
    }
}

/// Metadata record for an ML experiment run or strategy benchmark in the Spodeian Data Lake.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentRecord {
    pub run_id: String,
    pub experiment_name: String,
    pub config_yaml_snapshot: String,
    pub final_metrics: HashMap<String, f64>,
    pub checkpoint_path: Option<PathBuf>,
    pub checkpoint_sha256: Option<String>,
    pub hardware_context: HashMap<String, String>,
    pub timestamp_epoch_s: u64,
    pub duration_seconds: f64,
    pub completion_status: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Registry managing active dataset partitions, training utility, and value-aware eviction tracking.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LakeRegistry {
    partitions: HashMap<String, DatasetMetadata>,
    #[serde(default)]
    experiments: HashMap<String, ExperimentRecord>,
}

impl LakeRegistry {
    pub fn new() -> Self {
        Self {
            partitions: HashMap::new(),
            experiments: HashMap::new(),
        }
    }

    fn composite_key(dataset_id: &str, partition_key: &str) -> String {
        format!("{dataset_id}::{partition_key}")
    }

    /// Registers or updates a partition metadata record.
    pub fn register(&mut self, metadata: DatasetMetadata) {
        let key = Self::composite_key(&metadata.dataset_id, &metadata.partition_key);
        self.partitions.insert(key, metadata);
    }

    /// Retrieves metadata for a specific partition.
    pub fn get(&self, dataset_id: &str, partition_key: &str) -> Option<&DatasetMetadata> {
        let key = Self::composite_key(dataset_id, partition_key);
        self.partitions.get(&key)
    }

    /// Updates the last accessed timestamp for LRU tracking.
    pub fn record_access(
        &mut self,
        dataset_id: &str,
        partition_key: &str,
        current_epoch_s: u64,
    ) -> bool {
        let key = Self::composite_key(dataset_id, partition_key);
        if let Some(entry) = self.partitions.get_mut(&key) {
            entry.last_accessed_epoch_s = current_epoch_s;
            true
        } else {
            false
        }
    }

    /// Updates training utility, anomaly score, and epoch count from ML telemetry feedback.
    pub fn record_training_feedback(
        &mut self,
        dataset_id: &str,
        partition_key: &str,
        utility: f64,
        anomaly_score: Option<f64>,
        epochs_increment: usize,
    ) -> bool {
        let key = Self::composite_key(dataset_id, partition_key);
        if let Some(entry) = self.partitions.get_mut(&key) {
            entry.training_utility = utility.clamp(0.0, 1.0);
            if let Some(anomaly) = anomaly_score {
                entry.anomaly_score = anomaly.clamp(0.0, 1.0);
            }
            entry.training_epochs_seen += epochs_increment;
            true
        } else {
            false
        }
    }

    /// Pins or unpins a partition to protect or unprotect it from eviction.
    pub fn set_pinned(&mut self, dataset_id: &str, partition_key: &str, pinned: bool) -> bool {
        let key = Self::composite_key(dataset_id, partition_key);
        if let Some(entry) = self.partitions.get_mut(&key) {
            entry.is_pinned = pinned;
            true
        } else {
            false
        }
    }

    /// Unregisters and removes a partition, returning its metadata.
    pub fn unregister(&mut self, dataset_id: &str, partition_key: &str) -> Option<DatasetMetadata> {
        let key = Self::composite_key(dataset_id, partition_key);
        self.partitions.remove(&key)
    }

    /// Returns a list of partitions that have exceeded their TTL and have low training utility.
    pub fn get_stale_partitions(&self, current_epoch_s: u64) -> Vec<(String, String)> {
        self.partitions
            .values()
            .filter(|meta| meta.is_stale(current_epoch_s))
            .map(|meta| (meta.dataset_id.clone(), meta.partition_key.clone()))
            .collect()
    }

    /// Returns all partitions for a given dataset_id created within `[start_epoch, end_epoch]`, sorted chronologically.
    pub fn get_partitions_in_range(
        &self,
        dataset_id: &str,
        start_epoch: u64,
        end_epoch: u64,
    ) -> Vec<DatasetMetadata> {
        let mut list: Vec<DatasetMetadata> = self
            .partitions
            .values()
            .filter(|meta| {
                meta.dataset_id == dataset_id
                    && meta.created_at_epoch_s >= start_epoch
                    && meta.created_at_epoch_s <= end_epoch
            })
            .cloned()
            .collect();

        list.sort_by_key(|meta| meta.created_at_epoch_s);
        list
    }

    /// Returns partitions sorted in ascending order of retention priority (lowest priority first).
    /// Used for value-aware eviction when the data lake reaches capacity.
    pub fn get_eviction_candidates_by_priority(
        &self,
        current_epoch_s: u64,
    ) -> Vec<DatasetMetadata> {
        let mut list: Vec<DatasetMetadata> = self
            .partitions
            .values()
            .filter(|meta| !meta.is_pinned) // Never evict pinned partitions
            .cloned()
            .collect();

        list.sort_by(|a, b| {
            let score_a = a.compute_retention_priority(current_epoch_s);
            let score_b = b.compute_retention_priority(current_epoch_s);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        list
    }

    /// Returns all partitions sorted in ascending order of last accessed time (LRU order).
    pub fn get_lru_sorted(&self) -> Vec<DatasetMetadata> {
        let mut list: Vec<DatasetMetadata> = self.partitions.values().cloned().collect();
        list.sort_by_key(|meta| meta.last_accessed_epoch_s);
        list
    }

    /// Calculates total storage footprint of all registered partitions.
    pub fn total_bytes(&self) -> u64 {
        self.partitions.values().map(|meta| meta.byte_size).sum()
    }

    /// Total count of registered partitions.
    pub fn count(&self) -> usize {
        self.partitions.len()
    }

    /// Total count of records across all registered partitions.
    pub fn total_records(&self) -> usize {
        self.partitions.values().map(|meta| meta.record_count).sum()
    }

    /// Registers or updates an experiment record.
    pub fn register_experiment(&mut self, record: ExperimentRecord) {
        self.experiments.insert(record.run_id.clone(), record);
    }

    /// Retrieves an experiment record by run_id.
    pub fn get_experiment(&self, run_id: &str) -> Option<&ExperimentRecord> {
        self.experiments.get(run_id)
    }

    /// Lists all experiment records sorted by timestamp descending.
    pub fn list_experiments(&self) -> Vec<ExperimentRecord> {
        let mut list: Vec<ExperimentRecord> = self.experiments.values().cloned().collect();
        list.sort_by_key(|a| std::cmp::Reverse(a.timestamp_epoch_s));
        list
    }

    /// Lists experiment records by experiment_name sorted by timestamp descending.
    pub fn list_experiments_by_name(&self, experiment_name: &str) -> Vec<ExperimentRecord> {
        let mut list: Vec<ExperimentRecord> = self
            .experiments
            .values()
            .filter(|exp| exp.experiment_name == experiment_name)
            .cloned()
            .collect();
        list.sort_by_key(|a| std::cmp::Reverse(a.timestamp_epoch_s));
        list
    }

    /// Unregisters and removes an experiment record.
    pub fn remove_experiment(&mut self, run_id: &str) -> Option<ExperimentRecord> {
        self.experiments.remove(run_id)
    }

    /// Total count of registered experiments.
    pub fn experiment_count(&self) -> usize {
        self.experiments.len()
    }

    /// Persists registry snapshot to a JSON metadata file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), RegistryError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Restores registry from a JSON metadata file.
    pub fn load_from_file(path: &Path) -> Result<Self, RegistryError> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let content = std::fs::read_to_string(path)?;
        let registry: Self = serde_json::from_str(&content)?;
        Ok(registry)
    }

    /// Applies a WAL entry directly in-memory.
    pub fn apply_wal_entry(&mut self, entry: &LakeWalEntry) {
        match entry {
            LakeWalEntry::Register(meta) => self.register(meta.clone()),
            LakeWalEntry::Remove {
                dataset_id,
                partition_key,
            } => {
                self.unregister(dataset_id, partition_key);
            }
            LakeWalEntry::Feedback {
                dataset_id,
                partition_key,
                utility,
                anomaly_score,
                epochs_increment,
            } => {
                self.record_training_feedback(
                    dataset_id,
                    partition_key,
                    *utility,
                    *anomaly_score,
                    *epochs_increment,
                );
            }
            LakeWalEntry::Pin {
                dataset_id,
                partition_key,
                is_pinned,
            } => {
                self.set_pinned(dataset_id, partition_key, *is_pinned);
            }
            LakeWalEntry::RegisterExperiment(record) => {
                self.register_experiment(record.clone());
            }
            LakeWalEntry::RemoveExperiment { run_id } => {
                self.remove_experiment(run_id);
            }
        }
    }

    /// Appends a single WAL entry to the append-only log file.
    pub fn append_wal(wal_path: &Path, entry: &LakeWalEntry) -> Result<(), RegistryError> {
        use std::io::Write;
        if let Some(parent) = wal_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(wal_path)?;
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    /// Loads registry from snapshot and replays any trailing WAL entries.
    pub fn load_with_wal(snapshot_path: &Path, wal_path: &Path) -> Result<Self, RegistryError> {
        let mut registry = Self::load_from_file(snapshot_path)?;
        if wal_path.exists() {
            use std::io::{BufRead, BufReader};
            let file = std::fs::File::open(wal_path)?;
            let reader = BufReader::new(file);
            for line_res in reader.lines() {
                let line = line_res?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<LakeWalEntry>(&line) {
                    registry.apply_wal_entry(&entry);
                }
            }
        }
        Ok(registry)
    }

    /// Flushes and compacts WAL: saves snapshot and removes the WAL log file.
    pub fn compact_wal(&self, snapshot_path: &Path, wal_path: &Path) -> Result<(), RegistryError> {
        self.save_to_file(snapshot_path)?;
        if wal_path.exists() {
            let _ = std::fs::remove_file(wal_path);
        }
        Ok(())
    }
}

/// Append-only Write-Ahead Log (WAL) mutation entry for LakeRegistry.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LakeWalEntry {
    Register(DatasetMetadata),
    Remove {
        dataset_id: String,
        partition_key: String,
    },
    Feedback {
        dataset_id: String,
        partition_key: String,
        utility: f64,
        anomaly_score: Option<f64>,
        epochs_increment: usize,
    },
    Pin {
        dataset_id: String,
        partition_key: String,
        is_pinned: bool,
    },
    RegisterExperiment(ExperimentRecord),
    RemoveExperiment {
        run_id: String,
    },
}
