use candle_core::{Device, Result, safetensors::load};

use ai_candle::c06_smooth_and_physics_informed_ai::t03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/03_smoothness_regularization_sobolev_jacobian_and_shift_equivariance.safetensors",
        ),
        &device,
    )?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let grad_x = topic::sobolev_gradient(&w["x_feat"], &w["w1"], &w["b1"], &w["w2_scalar"])?;
    common::assert_max_diff(&grad_x, &w["sobolev_grad"], 1e-5)?;

    let j_sq = topic::jacobian_frobenius_sq(&w["x_feat"], &w["w1"], &w["b1"], &w["w2_multi"])?;
    common::assert_max_diff(&j_sq, &w["j_frob_sq"], 1e-5)?;

    let y_blur = topic::blur_downsample_1d(&w["x_signal"], &w["kernel_1d"])?;
    common::assert_max_diff(&y_blur, &w["y_blur"], 1e-5)?;

    Ok(())
}
