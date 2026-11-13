use candle_core::{Device, Result, safetensors::load};

use ai_candle::c10_xai_and_ai_analysis_methods::t06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. ROME
    let (rome_delta, rome_wnew) = topic::rome_rank_one_edit(
        &w["rome_w"],
        &w["rome_k_star"],
        &w["rome_v_star"],
        &w["rome_cov_inv"],
    )?;
    common::assert_max_diff(&rome_delta, &w["rome_delta_ref"], 1e-5)?;
    common::assert_max_diff(&rome_wnew, &w["rome_wnew_ref"], 1e-5)?;

    // 3. MEMIT
    let (memit_delta, memit_wnew) = topic::memit_mass_edit(
        &w["rome_w"],
        &w["memit_keys"],
        &w["memit_vals"],
        &w["rome_cov_inv"],
        1e-4,
    )?;
    common::assert_max_diff(&memit_delta, &w["memit_delta_ref"], 1e-5)?;
    common::assert_max_diff(&memit_wnew, &w["memit_wnew_ref"], 1e-5)?;

    // 4. Model Stitching
    let (stitch_s, stitch_bhat) =
        topic::model_stitching_layer(&w["stitch_a"], &w["stitch_b"], 1e-4)?;
    common::assert_max_diff(&stitch_s, &w["stitch_s_ref"], 1e-5)?;
    common::assert_max_diff(&stitch_bhat, &w["stitch_bhat_ref"], 1e-5)?;

    // 5. Neuron Knockout
    let ko_pruned = topic::neuron_knockout(&w["rome_w"], &w["ko_mask"])?;
    common::assert_max_diff(&ko_pruned, &w["ko_pruned_ref"], 1e-5)?;

    Ok(())
}
