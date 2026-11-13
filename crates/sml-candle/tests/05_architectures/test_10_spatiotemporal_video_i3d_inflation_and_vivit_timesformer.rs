use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t10_spatiotemporal_video_i3d_inflation_and_vivit_timesformer as video;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_spatiotemporal_video_i3d_inflation_and_vivit_timesformer_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors",
    );
    let weights = load(weights_path, &device)?;

    // 1. I3D Weight Inflation Verification
    let conv2d_w = weights.get("conv2d_w").unwrap();
    let expected_i3d_w = weights.get("i3d_w").unwrap();
    let inflated_w = video::i3d_inflate_weights(conv2d_w, 3)?;
    let diff_w = inflated_w
        .sub(expected_i3d_w)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_w < 1e-5, "I3D weight inflation error: {}", diff_w);

    // 2. 3D Conv Forward Verification
    let x_video = weights.get("x_video").unwrap();
    let i3d_b = weights.get("i3d_b").unwrap();
    let expected_y3d = weights.get("y_3d").unwrap();

    let out_3d = video::conv3d_forward(x_video, expected_i3d_w, i3d_b)?;
    let diff_3d = out_3d
        .sub(expected_y3d)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_3d < 1e-5, "3D Conv parity error: {}", diff_3d);

    // 3. TimeSformer Attention Verification
    let tokens = weights.get("tokens").unwrap();
    let spatial_q_w = weights.get("spatial_q_w").unwrap();
    let spatial_k_w = weights.get("spatial_k_w").unwrap();
    let spatial_v_w = weights.get("spatial_v_w").unwrap();
    let spatial_out_w = weights.get("spatial_out_w").unwrap();
    let temporal_q_w = weights.get("temporal_q_w").unwrap();
    let temporal_k_w = weights.get("temporal_k_w").unwrap();
    let temporal_v_w = weights.get("temporal_v_w").unwrap();
    let temporal_out_w = weights.get("temporal_out_w").unwrap();
    let expected_tf = weights.get("timesformer_output").unwrap();

    let out_tf = video::timesformer_attention_forward(
        tokens,
        spatial_q_w,
        spatial_k_w,
        spatial_v_w,
        spatial_out_w,
        temporal_q_w,
        temporal_k_w,
        temporal_v_w,
        temporal_out_w,
    )?;
    let diff_tf = out_tf
        .sub(expected_tf)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_tf < 1e-5, "TimeSformer parity error: {}", diff_tf);

    Ok(())
}
