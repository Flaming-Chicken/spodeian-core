use candle_core::{Device, Result, safetensors::load};

use ai_candle::c14_quantisation::t24_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_24_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/05_kv_attention_optim_and_benchmarks/04_quantized_gradients_qsgd_terngrad_and_1bit_adam_error_feedback.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // QSGD
    let (q_grad, q_norm) = topic::qsgd_quantize(&w["grad"], 7)?;
    common::assert_max_diff(&q_norm, &w["qsgd_norm"], 1e-5)?;
    common::assert_max_diff(&q_grad, &w["qsgd_grad"], 1e-5)?;

    // TernGrad
    let (t_grad, t_scale) = topic::terngrad_quantize(&w["grad"], 2.5)?;
    common::assert_max_diff(&t_scale, &w["terngrad_scale"], 1e-5)?;
    common::assert_max_diff(&t_grad, &w["terngrad_grad"], 1e-5)?;

    // 1-bit Adam step 1
    let (m_hat_1, err_1, s_1) = topic::onebit_adam_compress(&w["grad"], &w["error_0"])?;
    common::assert_max_diff(&s_1, &w["adam_scale_1"], 1e-5)?;
    common::assert_max_diff(&m_hat_1, &w["adam_m_hat_1"], 1e-5)?;
    common::assert_max_diff(&err_1, &w["adam_error_1"], 1e-5)?;

    // 1-bit Adam step 2
    let (m_hat_2, err_2, s_2) = topic::onebit_adam_compress(&w["grad_2"], &err_1)?;
    common::assert_max_diff(&s_2, &w["adam_scale_2"], 1e-5)?;
    common::assert_max_diff(&m_hat_2, &w["adam_m_hat_2"], 1e-5)?;
    common::assert_max_diff(&err_2, &w["adam_error_2"], 1e-5)?;

    // Update step
    let new_param =
        topic::onebit_adam_update_step(&w["param"], &m_hat_1, &w["v"], 1e-3, 1e-8, 0.01)?;
    common::assert_max_diff(&new_param, &w["new_param"], 1e-5)?;

    Ok(())
}
