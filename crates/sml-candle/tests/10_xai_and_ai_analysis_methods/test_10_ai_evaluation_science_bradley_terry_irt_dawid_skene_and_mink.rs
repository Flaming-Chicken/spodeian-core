use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Bradley-Terry-Luce win probabilities
    let btl_p = topic::btl_win_probability(&w["btl_skills"])?;
    common::assert_max_diff(&btl_p, &w["btl_win_prob_ref"], 1e-5)?;

    // 3. 2PL IRT item response probabilities
    let irt_p = topic::irt_2pl_probability(
        &w["irt_abilities"],
        &w["irt_difficulties"],
        &w["irt_discriminations"],
    )?;
    common::assert_max_diff(&irt_p, &w["irt_prob_ref"], 1e-5)?;

    // 4. Dawid-Skene E-step posterior
    let ds_post =
        topic::dawid_skene_e_step(&w["ds_annotations"], &w["ds_priors"], &w["ds_confusion"])?;
    common::assert_max_diff(&ds_post, &w["ds_posterior_ref"], 1e-5)?;

    // 5. Min-K% Prob score
    let mink = topic::min_k_percent_prob(&w["mink_log_probs"], 0.25)?;
    common::assert_max_diff(&mink, &w["mink_score_ref"], 1e-5)?;

    Ok(())
}
