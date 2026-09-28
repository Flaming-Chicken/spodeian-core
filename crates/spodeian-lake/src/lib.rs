//! Automated Windows Data Lake with LRU staleness eviction, metadata tracking,
//! and Apache Parquet streaming for the Spodeian AI ecosystem.

pub mod parquet_io;
pub mod registry;

use registry::{DatasetMetadata, LakeRegistry, RegistryError};
pub use parquet_io::{ParquetError, TabularDataset};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LakeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Registry error: {0}")]
    Registry(#[from] RegistryError),
    #[error("Parquet error: {0}")]
    Parquet(#[from] ParquetError),
    #[error("Partition not found: {0}:{1}")]
    PartitionNotFound(String, String),
    #[error("Integrity check failed: expected checksum {expected}, got {actual}")]
    IntegrityFailure { expected: String, actual: String },
}

/// Configuration options for the local Windows Data Lake.
#[derive(Debug, Clone)]
pub struct DataLakeConfig {
    pub lake_root: PathBuf,
    pub max_storage_bytes: u64,
    pub default_ttl_seconds: Option<u64>,
}

impl Default for DataLakeConfig {
    fn default() -> Self {
        let base_dir = std::env::var("LOCALAPPDATA")
            .map(|appdata| PathBuf::from(appdata).join("spodeian").join("lake"))
            .unwrap_or_else(|_| PathBuf::from(".spodeian").join("lake"));

        Self {
            lake_root: base_dir,
            max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            default_ttl_seconds: Some(30 * 86400),       // 30 days
        }
    }
}

/// High-level summary of Data Lake metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LakeSummary {
    pub total_partitions: usize,
    pub total_records: usize,
    pub total_bytes: u64,
    pub max_bytes: u64,
    pub utilization_percent: f64,
    pub root_path: String,
}

/// Local Windows Data Lake manager providing on-demand ingestion, Parquet streaming,
/// and automated LRU/TTL staleness eviction.
pub struct WindowsDataLake {
    config: DataLakeConfig,
    registry: LakeRegistry,
    registry_path: PathBuf,
}

impl WindowsDataLake {
    /// Initializes a data lake instance with custom configuration.
    pub fn new(config: DataLakeConfig) -> Result<Self, LakeError> {
        fs::create_dir_all(&config.lake_root)?;
        let data_dir = config.lake_root.join("data");
        fs::create_dir_all(&data_dir)?;

        let registry_path = config.lake_root.join("lake_registry.json");
        let registry = LakeRegistry::load_from_file(&registry_path)?;

        Ok(Self {
            config,
            registry,
            registry_path,
        })
    }

    /// Initializes a data lake using default Windows AppData paths.
    pub fn default_windows() -> Result<Self, LakeError> {
        Self::new(DataLakeConfig::default())
    }

    /// Returns the active configuration.
    pub fn config(&self) -> &DataLakeConfig {
        &self.config
    }

    /// Persists the registry to disk.
    fn persist_registry(&self) -> Result<(), LakeError> {
        self.registry.save_to_file(&self.registry_path)?;
        Ok(())
    }

    /// Computes a relative or absolute file path for a partition.
    fn partition_path(&self, dataset_id: &str, partition_key: &str) -> PathBuf {
        let sanitized_dataset = dataset_id.replace(['/', '\\', ':', '*'], "_");
        let sanitized_partition = partition_key.replace(['/', '\\', ':', '*'], "_");
        self.config
            .lake_root
            .join("data")
            .join(format!("{sanitized_dataset}_{sanitized_partition}.parquet"))
    }

    /// Stores a `TabularDataset` as a standard Apache Parquet partition.
    pub fn store_parquet(
        &mut self,
        dataset_id: &str,
        partition_key: &str,
        dataset: &TabularDataset,
        ttl_seconds: Option<u64>,
        current_epoch_s: u64,
    ) -> Result<DatasetMetadata, LakeError> {
        let file_path = self.partition_path(dataset_id, partition_key);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        self.store_parquet_with_options(dataset_id, partition_key, dataset, ttl_seconds, 0.0, false, current_epoch_s)
    }

    /// Stores a `TabularDataset` with explicit regime anomaly flags and crisis memory pinning.
    pub fn store_parquet_with_options(
        &mut self,
        dataset_id: &str,
        partition_key: &str,
        dataset: &TabularDataset,
        ttl_seconds: Option<u64>,
        anomaly_score: f64,
        is_pinned: bool,
        current_epoch_s: u64,
    ) -> Result<DatasetMetadata, LakeError> {
        let file_path = self.partition_path(dataset_id, partition_key);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Serialize Parquet in memory first to compute SHA-256 and byte size
        let mut buffer = Vec::new();
        dataset.write_to_parquet(&mut buffer)?;
        let byte_size = buffer.len() as u64;

        let mut hasher = Sha256::new();
        hasher.update(&buffer);
        let sha256_checksum = format!("{:x}", hasher.finalize());

        fs::write(&file_path, &buffer)?;

        let metadata = DatasetMetadata {
            dataset_id: dataset_id.to_string(),
            partition_key: partition_key.to_string(),
            file_path: file_path.clone(),
            record_count: dataset.num_rows,
            byte_size,
            sha256_checksum,
            created_at_epoch_s: current_epoch_s,
            last_accessed_epoch_s: current_epoch_s,
            ttl_seconds: ttl_seconds.or(self.config.default_ttl_seconds),
            training_utility: 1.0, // Freshly ingested data starts with maximal training utility
            anomaly_score: anomaly_score.clamp(0.0, 1.0),
            training_epochs_seen: 0,
            is_pinned,
        };

        self.registry.register(metadata.clone());
        self.persist_registry()?;

        // Automatically enforce maximum storage capacity if exceeded using value-aware priority
        self.enforce_capacity(current_epoch_s)?;

        Ok(metadata)
    }

    /// Ingests a batch of tabular dataset partitions in parallel using Rayon multicore threads.
    /// Concurrently serializes Parquet buffers, writes files to disk, and calculates SHA-256 hashes.
    pub fn store_parquet_batch_parallel(
        &mut self,
        items: &[(&str, &str, &TabularDataset, Option<u64>, f64, bool)],
        current_epoch_s: u64,
    ) -> Result<Vec<DatasetMetadata>, LakeError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        use rayon::prelude::*;

        // Prepare file paths and ensure parents exist
        let paths: Vec<PathBuf> = items
            .iter()
            .map(|(ds, part, ..)| self.partition_path(ds, part))
            .collect();

        for path in &paths {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
        }

        let default_ttl = self.config.default_ttl_seconds;

        // Concurrently serialize Parquet, calculate SHA-256 checksums, and write to disk
        let serialized_results: Result<Vec<DatasetMetadata>, LakeError> = items
            .par_iter()
            .zip(paths.par_iter())
            .map(|(&(ds_id, part_key, dataset, ttl, anomaly, pinned), file_path)| {
                let mut buffer = Vec::new();
                dataset.write_to_parquet(&mut buffer).map_err(LakeError::Parquet)?;
                let byte_size = buffer.len() as u64;

                let mut hasher = Sha256::new();
                hasher.update(&buffer);
                let sha256_checksum = format!("{:x}", hasher.finalize());

                fs::write(file_path, &buffer).map_err(LakeError::Io)?;

                Ok(DatasetMetadata {
                    dataset_id: ds_id.to_string(),
                    partition_key: part_key.to_string(),
                    file_path: file_path.clone(),
                    record_count: dataset.num_rows,
                    byte_size,
                    sha256_checksum,
                    created_at_epoch_s: current_epoch_s,
                    last_accessed_epoch_s: current_epoch_s,
                    ttl_seconds: ttl.or(default_ttl),
                    training_utility: 1.0,
                    anomaly_score: anomaly.clamp(0.0, 1.0),
                    training_epochs_seen: 0,
                    is_pinned: pinned,
                })
            })
            .collect();

        let metadata_list = serialized_results?;

        // Batch update registry
        for meta in &metadata_list {
            self.registry.register(meta.clone());
        }
        self.persist_registry()?;
        self.enforce_capacity(current_epoch_s)?;

        Ok(metadata_list)
    }

    /// Concurrently validates the SHA-256 integrity of multiple partitions using Rayon.
    /// Returns Ok(true) if all verified correctly, or LakeError on corruption or failure.
    pub fn verify_partitions_parallel(
        &self,
        partition_keys: &[(&str, &str)],
    ) -> Result<bool, LakeError> {
        if partition_keys.is_empty() {
            return Ok(true);
        }

        use rayon::prelude::*;

        let metas: Result<Vec<DatasetMetadata>, LakeError> = partition_keys
            .iter()
            .map(|&(ds_id, part_key)| {
                self.registry
                    .get(ds_id, part_key)
                    .cloned()
                    .ok_or_else(|| LakeError::PartitionNotFound(ds_id.to_string(), part_key.to_string()))
            })
            .collect();

        let metas = metas?;

        let verification_results: Result<Vec<()>, LakeError> = metas
            .par_iter()
            .map(|meta| {
                if !meta.file_path.exists() {
                    return Err(LakeError::PartitionNotFound(
                        meta.dataset_id.clone(),
                        meta.partition_key.clone(),
                    ));
                }
                let bytes = fs::read(&meta.file_path).map_err(LakeError::Io)?;
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                let actual = format!("{:x}", hasher.finalize());

                if actual != meta.sha256_checksum {
                    Err(LakeError::IntegrityFailure {
                        expected: meta.sha256_checksum.clone(),
                        actual,
                    })
                } else {
                    Ok(())
                }
            })
            .collect();

        verification_results.map(|_| true)
    }

    /// Records training feedback (loss/reconstruction error and exposure count) from ML training loops.
    pub fn record_training_feedback(
        &mut self,
        dataset_id: &str,
        partition_key: &str,
        utility: f64,
        anomaly_score: Option<f64>,
        epochs_increment: usize,
    ) -> Result<bool, LakeError> {
        let updated = self.registry.record_training_feedback(
            dataset_id,
            partition_key,
            utility,
            anomaly_score,
            epochs_increment,
        );
        if updated {
            self.persist_registry()?;
        }
        Ok(updated)
    }

    /// Pins or unpins a partition to protect it from automated pruning (e.g. historical crisis memories).
    pub fn set_pinned(&mut self, dataset_id: &str, partition_key: &str, pinned: bool) -> Result<bool, LakeError> {
        let updated = self.registry.set_pinned(dataset_id, partition_key, pinned);
        if updated {
            self.persist_registry()?;
        }
        Ok(updated)
    }

    /// Loads a `TabularDataset` partition from disk and updates LRU telemetry.
    pub fn load_parquet(
        &mut self,
        dataset_id: &str,
        partition_key: &str,
        current_epoch_s: u64,
    ) -> Result<TabularDataset, LakeError> {
        let meta = self
            .registry
            .get(dataset_id, partition_key)
            .cloned()
            .ok_or_else(|| LakeError::PartitionNotFound(dataset_id.to_string(), partition_key.to_string()))?;

        if !meta.file_path.exists() {
            return Err(LakeError::PartitionNotFound(
                dataset_id.to_string(),
                partition_key.to_string(),
            ));
        }

        // Verify SHA-256 integrity
        let file_bytes = fs::read(&meta.file_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        let actual_checksum = format!("{:x}", hasher.finalize());

        if actual_checksum != meta.sha256_checksum {
            return Err(LakeError::IntegrityFailure {
                expected: meta.sha256_checksum,
                actual: actual_checksum,
            });
        }

        let dataset = TabularDataset::read_from_parquet(&file_bytes)?;

        // Update LRU access time
        self.registry
            .record_access(dataset_id, partition_key, current_epoch_s);
        self.persist_registry()?;

        Ok(dataset)
    }

    /// Prunes stale partitions whose TTL has expired relative to `current_epoch_s`.
    pub fn prune_stale(&mut self, current_epoch_s: u64) -> Result<usize, LakeError> {
        let stale_keys = self.registry.get_stale_partitions(current_epoch_s);
        let mut evicted_count = 0;

        for (dataset_id, partition_key) in stale_keys {
            if let Some(meta) = self.registry.unregister(&dataset_id, &partition_key) {
                if meta.file_path.exists() {
                    let _ = fs::remove_file(&meta.file_path);
                }
                evicted_count += 1;
            }
        }

        if evicted_count > 0 {
            self.persist_registry()?;
        }
        Ok(evicted_count)
    }

    /// Prunes partitions whose training utility has decayed below `threshold` and are not crisis memories.
    pub fn prune_low_utility(&mut self, threshold: f64) -> Result<usize, LakeError> {
        let mut evicted_count = 0;
        let candidates: Vec<(String, String)> = self
            .registry
            .get_eviction_candidates_by_priority(0)
            .into_iter()
            .filter(|m| m.training_utility < threshold && !m.is_pinned && m.anomaly_score < 0.5)
            .map(|m| (m.dataset_id, m.partition_key))
            .collect();

        for (dataset_id, partition_key) in candidates {
            if let Some(removed) = self.registry.unregister(&dataset_id, &partition_key) {
                if removed.file_path.exists() {
                    let _ = fs::remove_file(&removed.file_path);
                }
                evicted_count += 1;
            }
        }

        if evicted_count > 0 {
            self.persist_registry()?;
        }
        Ok(evicted_count)
    }

    /// Evicts partitions when total storage exceeds `max_storage_bytes` using training retention priority.
    /// Lowest-value partitions (low loss, zero crisis indicators, high epoch saturation) are pruned first.
    pub fn enforce_capacity(&mut self, current_epoch_s: u64) -> Result<usize, LakeError> {
        let mut evicted_count = 0;
        let mut total = self.registry.total_bytes();

        if total <= self.config.max_storage_bytes {
            return Ok(0);
        }

        let candidates = self.registry.get_eviction_candidates_by_priority(current_epoch_s);
        for meta in candidates {
            if total <= self.config.max_storage_bytes {
                break;
            }
            if let Some(removed) = self.registry.unregister(&meta.dataset_id, &meta.partition_key) {
                if removed.file_path.exists() {
                    let _ = fs::remove_file(&removed.file_path);
                }
                total = total.saturating_sub(removed.byte_size);
                evicted_count += 1;
            }
        }

        if evicted_count > 0 {
            self.persist_registry()?;
        }
        Ok(evicted_count)
    }

    /// Explicitly deletes a specific partition.
    pub fn delete_partition(&mut self, dataset_id: &str, partition_key: &str) -> Result<bool, LakeError> {
        if let Some(removed) = self.registry.unregister(dataset_id, partition_key) {
            if removed.file_path.exists() {
                fs::remove_file(&removed.file_path)?;
            }
            self.persist_registry()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Produces a telemetry summary of current data lake capacity and utilization.
    pub fn summary(&self) -> LakeSummary {
        let total_bytes = self.registry.total_bytes();
        let max_bytes = self.config.max_storage_bytes;
        let utilization_percent = if max_bytes > 0 {
            (total_bytes as f64 / max_bytes as f64) * 100.0
        } else {
            0.0
        };

        LakeSummary {
            total_partitions: self.registry.count(),
            total_records: self.registry.total_records(),
            total_bytes,
            max_bytes,
            utilization_percent,
            root_path: self.config.lake_root.display().to_string(),
        }
    }
}
