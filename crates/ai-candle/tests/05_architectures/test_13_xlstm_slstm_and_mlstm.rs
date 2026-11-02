use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t13_xlstm_slstm_and_mlstm as xlstm;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_xlstm_slstm_and_mlstm_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm.safetensors",
    );
    let weights = load(weights_path, &device)?;

    // 1. Linear anchor verification
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let expected_linear = weights.get("test_output").unwrap();

    let out_linear = xlstm::linear_anchor(test_in, weight, bias)?;
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

    // 2. sLSTM Verification
    let x_seq = weights.get("x_seq").unwrap();
    let slstm_w_i = weights.get("slstm_w_i").unwrap();
    let slstm_w_f = weights.get("slstm_w_f").unwrap();
    let slstm_w_z = weights.get("slstm_w_z").unwrap();
    let slstm_w_o = weights.get("slstm_w_o").unwrap();
    let slstm_u_i = weights.get("slstm_u_i").unwrap();
    let slstm_u_f = weights.get("slstm_u_f").unwrap();
    let slstm_u_z = weights.get("slstm_u_z").unwrap();
    let slstm_u_o = weights.get("slstm_u_o").unwrap();
    let slstm_b_i = weights.get("slstm_b_i").unwrap();
    let slstm_b_f = weights.get("slstm_b_f").unwrap();
    let slstm_b_z = weights.get("slstm_b_z").unwrap();
    let slstm_b_o = weights.get("slstm_b_o").unwrap();

    let expected_slstm_seq = weights.get("slstm_seq_out").unwrap();
    let expected_slstm_c = weights.get("slstm_final_c").unwrap();
    let expected_slstm_n = weights.get("slstm_final_n").unwrap();

    let (slstm_seq, slstm_c, slstm_n) = xlstm::slstm_forward(
        x_seq, slstm_w_i, slstm_w_f, slstm_w_z, slstm_w_o, slstm_u_i, slstm_u_f, slstm_u_z,
        slstm_u_o, slstm_b_i, slstm_b_f, slstm_b_z, slstm_b_o,
    )?;

    let diff_slstm_seq = slstm_seq
        .sub(expected_slstm_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_slstm_c = slstm_c
        .sub(expected_slstm_c)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_slstm_n = slstm_n
        .sub(expected_slstm_n)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_slstm_seq < 1e-5, "sLSTM seq error: {}", diff_slstm_seq);
    assert!(diff_slstm_c < 1e-5, "sLSTM final c error: {}", diff_slstm_c);
    assert!(diff_slstm_n < 1e-5, "sLSTM final n error: {}", diff_slstm_n);

    // 3. mLSTM Verification
    let mlstm_w_q = weights.get("mlstm_w_q").unwrap();
    let mlstm_w_k = weights.get("mlstm_w_k").unwrap();
    let mlstm_w_v = weights.get("mlstm_w_v").unwrap();
    let mlstm_w_i = weights.get("mlstm_w_i").unwrap();
    let mlstm_w_f = weights.get("mlstm_w_f").unwrap();
    let mlstm_w_o = weights.get("mlstm_w_o").unwrap();
    let mlstm_b_q = weights.get("mlstm_b_q").unwrap();
    let mlstm_b_k = weights.get("mlstm_b_k").unwrap();
    let mlstm_b_v = weights.get("mlstm_b_v").unwrap();
    let mlstm_b_i = weights.get("mlstm_b_i").unwrap();
    let mlstm_b_f = weights.get("mlstm_b_f").unwrap();
    let mlstm_b_o = weights.get("mlstm_b_o").unwrap();

    let expected_mlstm_seq = weights.get("mlstm_seq_out").unwrap();
    let expected_mlstm_c = weights.get("mlstm_final_c").unwrap();
    let expected_mlstm_n = weights.get("mlstm_final_n").unwrap();

    let (mlstm_seq, mlstm_c, mlstm_n) = xlstm::mlstm_forward(
        x_seq, mlstm_w_q, mlstm_w_k, mlstm_w_v, mlstm_w_i, mlstm_w_f, mlstm_w_o, mlstm_b_q,
        mlstm_b_k, mlstm_b_v, mlstm_b_i, mlstm_b_f, mlstm_b_o,
    )?;

    let diff_mlstm_seq = mlstm_seq
        .sub(expected_mlstm_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_mlstm_c = mlstm_c
        .sub(expected_mlstm_c)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_mlstm_n = mlstm_n
        .sub(expected_mlstm_n)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_mlstm_seq < 1e-5, "mLSTM seq error: {}", diff_mlstm_seq);
    assert!(diff_mlstm_c < 1e-5, "mLSTM final c error: {}", diff_mlstm_c);
    assert!(diff_mlstm_n < 1e-5, "mLSTM final n error: {}", diff_mlstm_n);

    Ok(())
}
