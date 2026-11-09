use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. McNemar test statistic
    let mcn = topic::mcnemar_test_statistic(&w["mcnemar_b"], &w["mcnemar_c"])?;
    common::assert_max_diff(&mcn, &w["mcnemar_stat_ref"], 1e-5)?;

    // 3. Nadeau-Bengio corrected t-test
    let (t_stat, sigma) = topic::nadeau_bengio_corrected_t(&w["cv_diffs"], 800.0, 200.0)?;
    common::assert_max_diff(&t_stat, &w["nb_t_ref"], 1e-5)?;
    common::assert_max_diff(&sigma, &w["nb_sigma_ref"], 1e-5)?;

    // 4. Friedman rank test
    let (chi_f, avg_ranks) = topic::friedman_rank_statistic(&w["ranks"])?;
    common::assert_max_diff(&chi_f, &w["friedman_stat_ref"], 1e-5)?;
    common::assert_max_diff(&avg_ranks, &w["avg_ranks_ref"], 1e-5)?;

    // 5. Bayesian ROPE probabilities
    let rope_probs = topic::bayesian_rope_probabilities(&w["cv_diffs"], 800.0, 200.0, 0.01)?;
    common::assert_max_diff(&rope_probs, &w["rope_probs_ref"], 1e-5)?;

    Ok(())
}
