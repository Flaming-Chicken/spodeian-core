use candle_core::{Device, Result, safetensors::load};

use sml_candle::c07_domain_tasks::t09_time_series_patchtst_nbeats_chronos_and_probabilistic as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_time_series_patchtst_nbeats_chronos_and_probabilistic_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/09_time_series_patchtst_nbeats_chronos_and_probabilistic.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. RevIN
    let (norm_x, mean, stdev) =
        topic::revin_normalize(&w["revin_x"], &w["revin_gamma"], &w["revin_beta"], 1e-5)?;
    common::assert_max_diff(&norm_x, &w["revin_norm_x"], 1e-5)?;

    let denorm_y = topic::revin_denormalize(
        &w["norm_y"],
        &mean,
        &stdev,
        &w["revin_gamma"],
        &w["revin_beta"],
    )?;
    common::assert_max_diff(&denorm_y, &w["denorm_y_ref"], 1e-5)?;

    // 2. PatchTST patching
    let patches = topic::patchtst_patching(&w["revin_x"], 8, 4)?;
    common::assert_max_diff(&patches, &w["patches_ref"], 1e-5)?;

    // 3. N-BEATS block
    let (backcast, forecast) = topic::nbeats_block(
        &w["nbeats_x"],
        &w["w_fc1"],
        &w["b_fc1"],
        &w["w_fc2"],
        &w["b_fc2"],
        &w["w_theta_b"],
        &w["w_theta_f"],
        &w["basis_b"],
        &w["basis_f"],
    )?;
    common::assert_max_diff(&backcast, &w["backcast_ref"], 1e-5)?;
    common::assert_max_diff(&forecast, &w["forecast_ref"], 1e-5)?;

    // 4. Pinball loss
    let loss = topic::pinball_quantile_loss(&w["y_true"], &w["y_pred_quantiles"], &w["quantiles"])?;
    common::assert_max_diff(&loss, &w["pinball_loss_ref"], 1e-5)?;

    Ok(())
}
