use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t14_structured_state_space_models_s4_and_lru as s4;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_14_structured_state_space_models_s4_and_lru_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru.safetensors",
    );
    let weights = load(weights_path, &device)?;

    // 1. Linear anchor verification
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let expected_linear = weights.get("test_output").unwrap();

    let out_linear = s4::linear_anchor(test_in, weight, bias)?;
    let diff_linear = out_linear
        .sub(expected_linear)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error: {}",
        diff_linear
    );

    // 2. LRU Recurrent Forward Verification
    let x_seq = weights.get("x_seq").unwrap();
    let nu = weights.get("nu").unwrap();
    let theta = weights.get("theta").unwrap();
    let b_re = weights.get("b_re").unwrap();
    let b_im = weights.get("b_im").unwrap();
    let c_re = weights.get("c_re").unwrap();
    let c_im = weights.get("c_im").unwrap();
    let d_skip = weights.get("d_skip").unwrap();

    let expected_seq = weights.get("seq_out").unwrap();
    let expected_h_re = weights.get("final_h_re").unwrap();
    let expected_h_im = weights.get("final_h_im").unwrap();

    let (seq_out, h_re, h_im) =
        s4::lru_recurrent_forward(x_seq, nu, theta, b_re, b_im, c_re, c_im, d_skip)?;

    let diff_seq = seq_out
        .sub(expected_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_h_re = h_re
        .sub(expected_h_re)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_h_im = h_im
        .sub(expected_h_im)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_seq < 1e-5, "LRU seq error: {}", diff_seq);
    assert!(diff_h_re < 1e-5, "LRU h_re error: {}", diff_h_re);
    assert!(diff_h_im < 1e-5, "LRU h_im error: {}", diff_h_im);

    Ok(())
}
