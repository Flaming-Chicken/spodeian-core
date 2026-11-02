use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Neural collapse metrics
    let (nc1, nc2) = topic::neural_collapse_metrics(&w["nc_feats"], &w["nc_labels"], 3)?;
    common::assert_max_diff(&nc1, &w["nc1_ref"], 1e-5)?;
    common::assert_max_diff(&nc2, &w["nc2_ref"], 1e-5)?;

    // 3. Superposition interference
    let (gram, interf) = topic::superposition_interference(&w["sp_w"])?;
    common::assert_max_diff(&gram, &w["sp_gram_ref"], 1e-5)?;
    common::assert_max_diff(&interf, &w["sp_interf_ref"], 1e-5)?;

    // 4. WeightWatcher RMT power law alpha
    let alpha = topic::weightwatcher_alpha_hill(&w["ww_evals"], 0.5)?;
    common::assert_max_diff(&alpha, &w["ww_alpha_ref"], 1e-5)?;

    Ok(())
}
