use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t07_classic_to_modern_cnns_resnet_densenet_convnext as cnn;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_classic_to_modern_cnns_resnet_densenet_convnext_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();

    // 1. ConvNeXt-V2 Verification
    let dw_w = weights.get("dw_w").unwrap();
    let dw_b = weights.get("dw_b").unwrap();
    let pw1_w = weights.get("pw1_w").unwrap();
    let pw1_b = weights.get("pw1_b").unwrap();
    let gamma = weights.get("grn_gamma").unwrap();
    let beta = weights.get("grn_beta").unwrap();
    let pw2_w = weights.get("pw2_w").unwrap();
    let pw2_b = weights.get("pw2_b").unwrap();

    let expected_convnext = weights.get("convnext_output").unwrap();
    let out_convnext =
        cnn::convnext_v2_forward(x, dw_w, dw_b, pw1_w, pw1_b, gamma, beta, pw2_w, pw2_b)?;
    let diff_convnext = out_convnext
        .sub(expected_convnext)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_convnext < 1e-5,
        "ConvNeXt-V2 parity error: {}",
        diff_convnext
    );

    // 2. Pre-Activation ResNet Verification
    let res_w1 = weights.get("res_w1").unwrap();
    let res_b1 = weights.get("res_b1").unwrap();
    let res_w2 = weights.get("res_w2").unwrap();
    let res_b2 = weights.get("res_b2").unwrap();

    let expected_resnet = weights.get("resnet_output").unwrap();
    let out_resnet = cnn::preact_resnet_forward(x, res_w1, res_b1, res_w2, res_b2)?;
    let diff_resnet = out_resnet
        .sub(expected_resnet)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_resnet < 1e-5, "ResNet parity error: {}", diff_resnet);

    Ok(())
}
