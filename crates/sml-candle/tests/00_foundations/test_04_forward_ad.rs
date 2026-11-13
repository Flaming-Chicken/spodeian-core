//! Integration test for Topic 04: Forward AD and Functional Transforms (Candle Track).

use candle_core::{Device, Result, safetensors::load};
use sml_candle::c00_foundations::t04_forward_ad_and_functional_transforms_torch_func::ForwardAdJvp;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight = tensors.get("layers.weight").expect("layers.weight");
    let bias = tensors.get("layers.bias").expect("layers.bias");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let tangent_in = tensors.get("layers.tangent_in").expect("layers.tangent_in");
    let tangent_out = tensors
        .get("layers.tangent_out")
        .expect("layers.tangent_out");

    // 1. Forward AD (JVP) pass
    let (candle_out, candle_jvp) = ForwardAdJvp::linear_jvp(test_in, tangent_in, weight, bias)?;

    // 2. Parity against golden reference outputs
    let diff_primals = candle_out
        .sub(test_out)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_jvp = candle_jvp
        .sub(tangent_out)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(
        diff_primals < 1e-5,
        "Candle primals error {:.2e} exceeds 1e-5",
        diff_primals
    );
    assert!(
        diff_jvp < 1e-5,
        "Candle JVP error {:.2e} exceeds 1e-5",
        diff_jvp
    );

    // 3. Per-sample gradient shape verification
    let per_sample_dw = ForwardAdJvp::per_sample_weights_grad(test_in, &candle_out, test_out)?;
    assert_eq!(per_sample_dw.dims(), &[4, 3, 2]);

    println!(
        "Candle Topic 04 Parity: Primals err={:.2e}, JVP err={:.2e}",
        diff_primals, diff_jvp
    );

    Ok(())
}
