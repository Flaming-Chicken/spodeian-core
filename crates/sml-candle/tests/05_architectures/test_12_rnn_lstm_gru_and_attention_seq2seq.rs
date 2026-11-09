use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t12_rnn_lstm_gru_and_attention_seq2seq as rnn;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_rnn_lstm_gru_and_attention_seq2seq_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.safetensors",
    );
    let weights = load(weights_path, &device)?;

    // 1. Linear anchor verification
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let expected_linear = weights.get("test_output").unwrap();

    let out_linear = rnn::linear_anchor(test_in, weight, bias)?;
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

    // 2. LSTM Verification
    let x_seq = weights.get("x_seq").unwrap();
    let h0 = weights.get("h0").unwrap();
    let c0 = weights.get("c0").unwrap();
    let lstm_w_ih = weights.get("lstm_w_ih").unwrap();
    let lstm_w_hh = weights.get("lstm_w_hh").unwrap();
    let lstm_b_ih = weights.get("lstm_b_ih").unwrap();
    let lstm_b_hh = weights.get("lstm_b_hh").unwrap();

    let expected_lstm_seq = weights.get("lstm_seq_out").unwrap();
    let expected_lstm_h = weights.get("lstm_final_h").unwrap();
    let expected_lstm_c = weights.get("lstm_final_c").unwrap();

    let (lstm_seq, lstm_h, lstm_c) =
        rnn::lstm_forward(x_seq, h0, c0, lstm_w_ih, lstm_w_hh, lstm_b_ih, lstm_b_hh)?;

    let diff_lstm_seq = lstm_seq
        .sub(expected_lstm_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_lstm_h = lstm_h
        .sub(expected_lstm_h)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_lstm_c = lstm_c
        .sub(expected_lstm_c)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_lstm_seq < 1e-5, "LSTM seq error: {}", diff_lstm_seq);
    assert!(diff_lstm_h < 1e-5, "LSTM final h error: {}", diff_lstm_h);
    assert!(diff_lstm_c < 1e-5, "LSTM final c error: {}", diff_lstm_c);

    // 3. GRU Verification
    let gru_w_ih = weights.get("gru_w_ih").unwrap();
    let gru_w_hh = weights.get("gru_w_hh").unwrap();
    let gru_b_ih = weights.get("gru_b_ih").unwrap();
    let gru_b_hh = weights.get("gru_b_hh").unwrap();

    let expected_gru_seq = weights.get("gru_seq_out").unwrap();
    let expected_gru_h = weights.get("gru_final_h").unwrap();

    let (gru_seq, gru_h) = rnn::gru_forward(x_seq, h0, gru_w_ih, gru_w_hh, gru_b_ih, gru_b_hh)?;

    let diff_gru_seq = gru_seq
        .sub(expected_gru_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_gru_h = gru_h
        .sub(expected_gru_h)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_gru_seq < 1e-5, "GRU seq error: {}", diff_gru_seq);
    assert!(diff_gru_h < 1e-5, "GRU final h error: {}", diff_gru_h);

    // 4. Luong Attention Verification
    let decoder_query = weights.get("decoder_query").unwrap();
    let w_attn = weights.get("w_attn").unwrap();
    let expected_context = weights.get("context").unwrap();
    let expected_attn_w = weights.get("attn_weights").unwrap();

    let (context, attn_w) = rnn::luong_attention_forward(decoder_query, expected_lstm_seq, w_attn)?;

    let diff_ctx = context
        .sub(expected_context)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_attn = attn_w
        .sub(expected_attn_w)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_ctx < 1e-5, "Attention context error: {}", diff_ctx);
    assert!(diff_attn < 1e-5, "Attention weights error: {}", diff_attn);

    Ok(())
}
