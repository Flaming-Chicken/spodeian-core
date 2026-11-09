//! Integration test for Category 03 Topic 13: Differentiable Sorting, Ranking, Top-k, Permutahedron, and I-MLE (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t13_differentiable_sorting_ranking_topk_permutahedron_and_imle::{
    diff_topk_forward, neural_sort_forward, soft_histogram_forward, soft_quantile_forward,
    soft_sort_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/13_differentiable_sorting_ranking_topk_permutahedron_and_imle.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let softsort_scores = tensors.get("softsort_scores").expect("softsort_scores");
    let expected_p_soft = tensors.get("expected_p_soft").expect("expected_p_soft");
    let expected_sorted_approx = tensors
        .get("expected_sorted_approx")
        .expect("expected_sorted_approx");

    let neuralsort_scores = tensors.get("neuralsort_scores").expect("neuralsort_scores");
    let expected_p_neural = tensors.get("expected_p_neural").expect("expected_p_neural");

    let topk_scores = tensors.get("topk_scores").expect("topk_scores");
    let expected_topk_weights = tensors
        .get("expected_topk_weights")
        .expect("expected_topk_weights");

    let hist_x = tensors.get("hist_x").expect("hist_x");
    let bin_centers = tensors.get("bin_centers").expect("bin_centers");
    let expected_histogram = tensors
        .get("expected_histogram")
        .expect("expected_histogram");

    let quantile_scores = tensors.get("quantile_scores").expect("quantile_scores");
    let expected_median = tensors.get("expected_median").expect("expected_median");

    // 1. Linear Anchor Parity
    let anchor = LinearAnchor::new(weight.clone(), bias.clone());
    let linear_pred = anchor.forward(test_in)?;
    let diff_linear = linear_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error {:.2e} exceeds 1e-5",
        diff_linear
    );

    // 2. SoftSort Parity
    let (p_soft_pred, sorted_pred) = soft_sort_forward(softsort_scores, 0.2)?;
    let diff_p_soft = p_soft_pred
        .sub(expected_p_soft)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_sorted = sorted_pred
        .sub(expected_sorted_approx)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_p_soft < 1e-5,
        "SoftSort P parity error {:.2e} exceeds 1e-5",
        diff_p_soft
    );
    assert!(
        diff_sorted < 1e-5,
        "SoftSort sorted approx parity error {:.2e} exceeds 1e-5",
        diff_sorted
    );

    // 3. NeuralSort Parity
    let p_neural_pred = neural_sort_forward(neuralsort_scores, 0.5)?;
    let diff_p_neural = p_neural_pred
        .sub(expected_p_neural)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_p_neural < 1e-5,
        "NeuralSort P parity error {:.2e} exceeds 1e-5",
        diff_p_neural
    );

    // 4. Diff Top-k Parity
    let topk_pred = diff_topk_forward(topk_scores, 2, 0.2, 15)?;
    let diff_topk = topk_pred
        .sub(expected_topk_weights)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_topk < 1e-5,
        "Diff Top-k parity error {:.2e} exceeds 1e-5",
        diff_topk
    );

    // 5. Soft Histogram Parity
    let hist_pred = soft_histogram_forward(hist_x, bin_centers, 1.0, 0.1)?;
    let diff_hist = hist_pred
        .sub(expected_histogram)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_hist < 1e-5,
        "Soft histogram parity error {:.2e} exceeds 1e-5",
        diff_hist
    );

    // 6. Soft Quantile Parity
    let median_pred = soft_quantile_forward(quantile_scores, 0.5, 0.2)?;
    let diff_median = median_pred
        .sub(expected_median)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_median < 1e-5,
        "Soft quantile parity error {:.2e} exceeds 1e-5",
        diff_median
    );

    Ok(())
}
