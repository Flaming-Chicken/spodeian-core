use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t07_active_learning_bald_coreset_and_badge as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_active_learning_bald_coreset_and_badge_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. BALD
    let bald = topic::bald_score(&w["mc_probs"], 1e-12)?;
    common::assert_max_diff(&bald, &w["bald_score_ref"], 1e-5)?;

    // 3. Core-Set
    let min_dists = topic::coreset_min_distances(&w["unlabeled_feats"], &w["labeled_feats"])?;
    common::assert_max_diff(&min_dists, &w["coreset_min_dists_ref"], 1e-5)?;

    // 4. BADGE
    let badge = topic::badge_gradient_embeddings(&w["feats"], &w["probs"])?;
    common::assert_max_diff(&badge, &w["badge_embeds_ref"], 1e-5)?;

    Ok(())
}
