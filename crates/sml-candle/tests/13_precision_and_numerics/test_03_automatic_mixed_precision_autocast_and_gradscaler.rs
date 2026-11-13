use candle_core::{Device, Result, safetensors::load};

use ai_candle::c13_precision_and_numerics::t03_automatic_mixed_precision_autocast_and_gradscaler as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_automatic_mixed_precision_autocast_and_gradscaler_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. AMP Autocast Matmul + Softmax
    let probs = topic::amp_autocast_matmul_softmax(&w["amp_x"], &w["amp_w"], &w["amp_b"])?;
    common::assert_max_diff(&probs, &w["amp_probs_ref"], 1e-5)?;

    // 3. Dynamic Scaler: Clean step
    let (unscaled_c, scale_c, tracker_c, skipped_c) =
        topic::dynamic_grad_scaler_step(&w["clean_grads"], 65536.0, 3, 2.0, 0.5, 4)?;
    common::assert_max_diff(&unscaled_c, &w["unscaled_clean_ref"], 1e-5)?;
    assert!((scale_c - w["scale_clean_ref"].to_scalar::<f32>()?).abs() < 1e-5);
    assert_eq!(
        tracker_c,
        w["tracker_clean_ref"].to_scalar::<i64>()? as usize
    );
    assert_eq!(skipped_c, w["skipped_clean_ref"].to_scalar::<f32>()? > 0.5);

    // 4. Dynamic Scaler: Overflow step
    let (_unscaled_o, scale_o, tracker_o, skipped_o) = topic::dynamic_grad_scaler_step(
        &w["unscaled_over_ref"].affine(65536.0, 0.0)?,
        65536.0,
        2,
        2.0,
        0.5,
        4,
    )?;
    assert!((scale_o - w["scale_over_ref"].to_scalar::<f32>()?).abs() < 1e-5);
    assert_eq!(
        tracker_o,
        w["tracker_over_ref"].to_scalar::<i64>()? as usize
    );
    assert_eq!(skipped_o, w["skipped_over_ref"].to_scalar::<f32>()? > 0.5);

    // 5. Unscale and Clip Gradients
    let (clipped, norm) =
        topic::unscale_and_clip_grads(&w["clip_scaled_grads"], 1000.0, 1.0, 1e-6)?;
    common::assert_max_diff(&clipped, &w["clipped_ref"], 1e-5)?;
    assert!((norm - w["norm_ref"].to_scalar::<f32>()?).abs() < 1e-5);

    Ok(())
}
