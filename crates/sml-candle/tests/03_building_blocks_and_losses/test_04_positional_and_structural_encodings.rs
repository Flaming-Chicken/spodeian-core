//! Integration test for Category 03 Topic 04: Positional and Structural Encodings (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t04_positional_and_structural_encodings::{
    alibi_forward, rope_forward, rwpe_forward, sinusoidal_pe_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/04_positional_and_structural_encodings.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");
    let pe = tensors.get("sinusoidal_pe").expect("sinusoidal_pe");
    let cos = tensors.get("cos_rope").expect("cos_rope");
    let sin = tensors.get("sin_rope").expect("sin_rope");
    let slopes = tensors.get("alibi_slopes").expect("alibi_slopes");
    let adj = tensors.get("graph_adj").expect("graph_adj");

    // 1. Linear Anchor Parity
    let anchor = LinearAnchor::new(weight.clone(), bias.clone());
    let linear_pred = anchor.forward(test_in)?;
    let diff_linear = linear_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error {:.2e} exceeds 1e-5",
        diff_linear
    );

    // 2. Sinusoidal PE Parity
    let y_sin_gold = tensors.get("y_sinusoidal").expect("y_sinusoidal");
    let sin_pred = sinusoidal_pe_forward(test_in, pe)?;
    let diff_sin = sin_pred
        .sub(y_sin_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sin < 1e-5,
        "Sinusoidal PE parity error {:.2e} exceeds 1e-5",
        diff_sin
    );

    // 3. RoPE Parity
    let y_rp_gold = tensors.get("y_rope").expect("y_rope");
    let rp_pred = rope_forward(test_in, cos, sin)?;
    let diff_rp = rp_pred
        .sub(y_rp_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rp < 1e-5,
        "RoPE parity error {:.2e} exceeds 1e-5",
        diff_rp
    );

    // 4. ALiBi Parity
    let y_alibi_gold = tensors.get("alibi_bias").expect("alibi_bias");
    let alibi_pred = alibi_forward(test_in.dim(0)?, slopes, &device)?;
    let diff_alibi = alibi_pred
        .sub(y_alibi_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_alibi < 1e-5,
        "ALiBi parity error {:.2e} exceeds 1e-5",
        diff_alibi
    );

    // 5. RWPE Parity
    let y_rw_gold = tensors.get("y_rwpe").expect("y_rwpe");
    let rw_pred = rwpe_forward(adj, 4)?;
    let diff_rw = rw_pred
        .sub(y_rw_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rw < 1e-5,
        "RWPE parity error {:.2e} exceeds 1e-5",
        diff_rw
    );

    Ok(())
}
