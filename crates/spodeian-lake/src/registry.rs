use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
}

impl DatasetMetadata {
    /// Determines whether the partition has exceeded its time-to-live.
    pub fn is_stale(&self, current_epoch_s: u64) -> bool {
        if let Some(ttl) = self.ttl_seconds {
            current_epoch_s > self.created_at_epoch_s.saturating_add(ttl)
        } else {
            false
        }
    }
}

/// Registry managing active dataset partitions, staleness detection, and LRU eviction tracking.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LakeRegistry {
    partitions: HashMap<String, DatasetMetadata>,
}

impl LakeRegistry {
    pub fn new() -> Self {
        Self {
            partitions: HashMap::new(),
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
    pub fn record_access(&mut self, dataset_id: &str, partition_key: &str, current_epoch_s: u64) -> bool {
        let key = Self::composite_key(dataset_id, partition_key);
        if let Some(entry) = self.partitions.get_mut(&key) {
            entry.last_accessed_epoch_s = current_epoch_s;
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

    /// Returns a list of partitions that have exceeded their TTL.
    pub fn get_stale_partitions(&self, current_epoch_s: u64) -> Vec<(String, String)> {
        self.partitions
            .values()
            .filter(|meta| meta.is_stale(current_epoch_s))
            .map(|meta| (meta.dataset_id.clone(), meta.partition_key.clone()))
            .collect()
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
}
