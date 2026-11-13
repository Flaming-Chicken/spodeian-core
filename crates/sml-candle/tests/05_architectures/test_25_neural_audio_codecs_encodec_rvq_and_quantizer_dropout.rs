use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t25_neural_audio_codecs_encodec_rvq_and_quantizer_dropout as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_25_neural_audio_codecs_encodec_rvq_and_quantizer_dropout_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Snake activation
    let alpha = w["alpha"].to_vec1::<f32>()?[0] as f64;
    let snake_out = topic::snake_activation(&w["x_audio"], alpha)?;
    common::assert_max_diff(&snake_out, &w["snake_out"], 1e-5)?;

    // 3. RVQ with quantizer dropout
    let n_active = w["n_active"].to_vec1::<i64>()?[0] as usize;
    let (z_q, indices, residual) =
        topic::residual_vector_quantization(&w["z_in"], &w["codebooks"], n_active)?;
    common::assert_max_diff(&z_q, &w["z_q"], 1e-5)?;
    common::assert_max_diff(&indices, &w["indices"], 1e-5)?;
    common::assert_max_diff(&residual, &w["residual"], 1e-5)
}
