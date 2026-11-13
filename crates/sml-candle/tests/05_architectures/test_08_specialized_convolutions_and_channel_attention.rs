use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t08_specialized_convolutions_and_channel_attention as spec_conv;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_specialized_convolutions_and_channel_attention_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();

    // 1. CoordConv Verification
    let coord_w = weights.get("coord_conv_w").unwrap();
    let coord_b = weights.get("coord_conv_b").unwrap();
    let expected_coord = weights.get("coord_output").unwrap();

    let out_coord = spec_conv::coord_conv2d(x, coord_w, coord_b)?;
    let diff_coord = out_coord
        .sub(expected_coord)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_coord < 1e-5, "CoordConv parity error: {}", diff_coord);

    // 2. Squeeze-and-Excitation Verification
    let se_fc1_w = weights.get("se_fc1_w").unwrap();
    let se_fc1_b = weights.get("se_fc1_b").unwrap();
    let se_fc2_w = weights.get("se_fc2_w").unwrap();
    let se_fc2_b = weights.get("se_fc2_b").unwrap();
    let expected_se = weights.get("se_output").unwrap();

    let out_se = spec_conv::squeeze_and_excitation(x, se_fc1_w, se_fc1_b, se_fc2_w, se_fc2_b)?;
    let diff_se = out_se
        .sub(expected_se)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_se < 1e-5, "SE parity error: {}", diff_se);

    // 3. Efficient Channel Attention (ECA) Verification
    let eca_w = weights.get("eca_conv_w").unwrap();
    let eca_b = weights.get("eca_conv_b").unwrap();
    let expected_eca = weights.get("eca_output").unwrap();

    let out_eca = spec_conv::efficient_channel_attention(x, eca_w, eca_b)?;
    let diff_eca = out_eca
        .sub(expected_eca)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_eca < 1e-5, "ECA parity error: {}", diff_eca);

    Ok(())
}
