use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t01_residual_mlp_mlp_mixer_and_gmlp as arch;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_residual_mlp_mlp_mixer_and_gmlp_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("x").unwrap();

    // 1. Verify MLP-Mixer
    let mixer_out = arch::mlp_mixer_forward(
        x,
        weights.get("w1_token").unwrap(),
        weights.get("b1_token").unwrap(),
        weights.get("w2_token").unwrap(),
        weights.get("b2_token").unwrap(),
        weights.get("w1_chan").unwrap(),
        weights.get("b1_chan").unwrap(),
        weights.get("w2_chan").unwrap(),
        weights.get("b2_chan").unwrap(),
        weights.get("ln1_w").unwrap(),
        weights.get("ln1_b").unwrap(),
        weights.get("ln2_w").unwrap(),
        weights.get("ln2_b").unwrap(),
        1e-5,
    )?;
    let expected_mixer = weights.get("mixer_out").unwrap();
    let diff_mixer = mixer_out
        .sub(expected_mixer)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_mixer < 1e-5, "MLP-Mixer error: {}", diff_mixer);

    // 2. Verify gMLP
    let gmlp_out = arch::gmlp_forward(
        x,
        weights.get("w_in").unwrap(),
        weights.get("b_in").unwrap(),
        weights.get("w_sgu").unwrap(),
        weights.get("b_sgu").unwrap(),
        weights.get("w_out").unwrap(),
        weights.get("b_out").unwrap(),
        weights.get("gmlp_ln_w").unwrap(),
        weights.get("gmlp_ln_b").unwrap(),
        1e-5,
    )?;
    let expected_gmlp = weights.get("gmlp_out").unwrap();
    let diff_gmlp = gmlp_out
        .sub(expected_gmlp)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_gmlp < 1e-5, "gMLP error: {}", diff_gmlp);

    // 3. Verify Anchor Parity
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let anchor_out = arch::linear_anchor_forward(test_in, weight, bias)?;
    let expected_anchor = weights.get("test_output").unwrap();
    let diff_anchor = anchor_out
        .sub(expected_anchor)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "Anchor error: {}", diff_anchor);

    Ok(())
}
