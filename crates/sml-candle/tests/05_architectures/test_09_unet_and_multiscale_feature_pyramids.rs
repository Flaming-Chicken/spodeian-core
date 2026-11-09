use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t09_unet_and_multiscale_feature_pyramids as unet;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_unet_and_multiscale_feature_pyramids_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();

    // 1. U-Net Verification
    let enc_w = weights.get("enc_w").unwrap();
    let enc_b = weights.get("enc_b").unwrap();
    let bot_w = weights.get("bot_w").unwrap();
    let bot_b = weights.get("bot_b").unwrap();
    let dec_w = weights.get("dec_w").unwrap();
    let dec_b = weights.get("dec_b").unwrap();
    let expected_unet = weights.get("unet_output").unwrap();

    let out_unet = unet::unet_mini_forward(x, enc_w, enc_b, bot_w, bot_b, dec_w, dec_b)?;
    let diff_unet = out_unet
        .sub(expected_unet)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_unet < 1e-5, "U-Net parity error: {}", diff_unet);

    // 2. BiFPN Fusion Verification
    let bifpn_w = weights.get("bifpn_w").unwrap();
    let bifpn_conv_w = weights.get("bifpn_conv_w").unwrap();
    let bifpn_conv_b = weights.get("bifpn_conv_b").unwrap();
    let expected_bifpn = weights.get("bifpn_output").unwrap();

    let out_bifpn = unet::bifpn_fusion(x, &out_unet, bifpn_w, bifpn_conv_w, bifpn_conv_b, 1e-4)?;
    let diff_bifpn = out_bifpn
        .sub(expected_bifpn)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_bifpn < 1e-5, "BiFPN parity error: {}", diff_bifpn);

    Ok(())
}
