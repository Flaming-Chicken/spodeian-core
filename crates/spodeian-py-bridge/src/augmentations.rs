//! Data augmentations re-exported from centralized spodeian-ml-utils.

pub use spodeian_ml_utils::factors::{
    apply_gaussian_jitter, apply_temporal_mask, rolling_zscore_normalize, time_warp_resample,
};
