//! Data augmentations re-exported from sml-runtime.

pub use sml_runtime::factors::{
    apply_gaussian_jitter, apply_temporal_mask, rolling_zscore_normalize, time_warp_resample,
};
