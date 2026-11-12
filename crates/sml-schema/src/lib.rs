//! Shared serialization data models for multi-track parameter metadata and architecture descriptors.

use serde::{Deserialize, Serialize};

/// Metadata descriptor for a synchronized AI-Examples topic across all tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicMetadata {
    pub category: String,
    pub slug: String,
    pub title: String,
    pub d_model: usize,
    pub seq_len: usize,
    pub parameter_count: usize,
    pub memory_bound: bool,
    pub parity_tolerance: f64,
}

/// Description of a tensor parameter buffer stored in safetensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub byte_size: usize,
}

/// Golden test vector manifest accompanying safetensors reference weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenVectorManifest {
    pub topic_slug: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub seed: u64,
    pub parameters: Vec<ParameterDescriptor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_metadata_serialization() {
        let meta = TopicMetadata {
            category: "00_foundations".to_string(),
            slug: "01_linear_neuron".to_string(),
            title: "Linear Neuron and Invertible Transforms".to_string(),
            d_model: 64,
            seq_len: 16,
            parameter_count: 65,
            memory_bound: true,
            parity_tolerance: 1e-5,
        };

        let json = serde_json::to_string(&meta).expect("Failed to serialize");
        let deserialized: TopicMetadata =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(meta, deserialized);
    }
}
