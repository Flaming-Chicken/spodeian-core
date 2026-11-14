//! Generic serialization descriptors for parameter buffers and golden parity test manifests.

use serde::{Deserialize, Serialize};

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
    fn test_golden_vector_manifest_serialization() {
        let manifest = GoldenVectorManifest {
            topic_slug: "00_foundations/01_linear_neuron".to_string(),
            input_shape: vec![8, 4],
            output_shape: vec![8, 4],
            seed: 42,
            parameters: vec![ParameterDescriptor {
                name: "weight".to_string(),
                shape: vec![4, 4],
                dtype: "F32".to_string(),
                byte_size: 64,
            }],
        };

        let json = serde_json::to_string(&manifest).expect("Failed to serialize");
        let deserialized: GoldenVectorManifest =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(manifest, deserialized);
    }
}
