use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t04_calibration_observers_minmax_percentile_mse_kl_and_aciq as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_calibration_observers_minmax_percentile_mse_kl_and_aciq_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/01_foundations_formats_and_observers/04_calibration_observers_minmax_percentile_mse_kl_and_aciq.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. MinMax Observer
    let (minmax_deq, minmax_s) = topic::minmax_observer(&w["raw_act"], 7.0)?;
    common::assert_max_diff(&minmax_deq, &w["minmax_dequant"], 1e-5)?;
    common::assert_max_diff(&minmax_s, &w["minmax_stats"], 1e-5)?;

    // 3. Moving Average MinMax
    let mov_s = topic::moving_average_minmax_observer(&w["raw_batches"], 0.1)?;
    common::assert_max_diff(&mov_s, &w["moving_avg_stats"], 1e-5)?;

    // 4. Percentile Observer
    let (perc_deq, perc_a) = topic::percentile_observer(&w["raw_act"], 0.95, 7.0)?;
    common::assert_max_diff(&perc_deq, &w["percentile_dequant"], 1e-5)?;
    common::assert_max_diff(&perc_a, &w["percentile_alpha"], 1e-5)?;

    // 5. MSE Observer
    let (mse_deq, mse_a) = topic::mse_observer(&w["raw_act"], 10, 7.0)?;
    common::assert_max_diff(&mse_deq, &w["mse_dequant"], 1e-5)?;
    common::assert_max_diff(&mse_a, &w["mse_alpha"], 1e-5)?;

    // 6. KL Divergence Observer
    let (kl_deq, kl_a) = topic::kl_divergence_observer(&w["raw_act"], 64, 8, 7.0)?;
    common::assert_max_diff(&kl_deq, &w["kl_dequant"], 1e-5)?;
    common::assert_max_diff(&kl_a, &w["kl_alpha"], 1e-5)?;

    // 7. ACIQ Gaussian
    let (ag_deq, ag_a) = topic::aciq_gaussian_observer(&w["raw_act"], 7.0)?;
    common::assert_max_diff(&ag_deq, &w["aciq_gauss_dequant"], 1e-5)?;
    common::assert_max_diff(&ag_a, &w["aciq_gauss_alpha"], 1e-5)?;

    // 8. ACIQ Laplace
    let (al_deq, al_a) = topic::aciq_laplace_observer(&w["raw_laplace"], 7.0)?;
    common::assert_max_diff(&al_deq, &w["aciq_laplace_dequant"], 1e-5)?;
    common::assert_max_diff(&al_a, &w["aciq_laplace_alpha"], 1e-5)?;

    Ok(())
}
