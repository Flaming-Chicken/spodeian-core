use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t05_mechanistic_interpretability_probing_lens_patching_and_sae as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_mechanistic_interpretability_probing_lens_patching_and_sae_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Logit Lens
    let logits = topic::logit_lens(&w["ll_h"], &w["ll_w"], &w["ll_b"])?;
    common::assert_max_diff(&logits, &w["ll_logits_ref"], 1e-5)?;

    // 3. Attribution Patching
    let atp = topic::attribution_patching(&w["atp_clean"], &w["atp_corrupt"], &w["atp_grads"])?;
    common::assert_max_diff(&atp, &w["atp_scores_ref"], 1e-5)?;

    // 4. Top-K SAE
    let (f_z, x_hat) = topic::topk_sparse_autoencoder(
        &w["sae_x"],
        &w["sae_w_enc"],
        &w["sae_b_enc"],
        &w["sae_w_dec"],
        &w["sae_b_dec"],
        2,
    )?;
    common::assert_max_diff(&f_z, &w["sae_feat_ref"], 1e-5)?;
    common::assert_max_diff(&x_hat, &w["sae_xhat_ref"], 1e-5)?;

    // 5. JumpReLU
    let jump = topic::jumprelu_activation(&w["jump_z"], &w["jump_thresh"])?;
    common::assert_max_diff(&jump, &w["jump_out_ref"], 1e-5)?;

    Ok(())
}
