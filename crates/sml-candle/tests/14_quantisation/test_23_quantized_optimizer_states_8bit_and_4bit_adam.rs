use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t23_quantized_optimizer_states_8bit_and_4bit_adam as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_23_quantized_optimizer_states_8bit_and_4bit_adam_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("14_quantisation/05_kv_attention_optim_and_benchmarks/03_quantized_optimizer_states_8bit_and_4bit_adam.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. 8-bit momentum quantization parity
    let (m_deq8_init, m_int8_init, scale_m8) = topic::quantize_8bit_first_moment(&w["m_prev"], 16)?;
    common::assert_max_diff(&m_deq8_init, &w["m_deq8_init"], 1e-5)?;
    common::assert_max_diff(&m_int8_init, &w["m_int8_init"], 1e-5)?;
    common::assert_max_diff(&scale_m8, &w["scale_m8"], 1e-5)?;

    let (v_deq8_init, v_int8_init, max_v8) = topic::quantize_8bit_second_moment(&w["v_prev"], 16)?;
    common::assert_max_diff(&v_deq8_init, &w["v_deq8_init"], 1e-5)?;
    common::assert_max_diff(&v_int8_init, &w["v_int8_init"], 1e-5)?;
    common::assert_max_diff(&max_v8, &w["max_v8"], 1e-5)?;

    // 3. 8-bit AdamW step parity
    let (theta_next_8bit, m_next8, v_next8) = topic::adamw_update_step(
        &w["theta"],
        &w["grad"],
        &w["m_deq8_init"],
        &w["v_deq8_init"],
        0.9,
        0.999,
        1e-3,
        1e-2,
        1,
    )?;
    common::assert_max_diff(&theta_next_8bit, &w["theta_next_8bit"], 1e-5)?;

    let (m_deq8_next, _, _) = topic::quantize_8bit_first_moment(&m_next8, 16)?;
    common::assert_max_diff(&m_deq8_next, &w["m_deq8_next"], 1e-5)?;

    let (v_deq8_next, _, _) = topic::quantize_8bit_second_moment(&v_next8, 16)?;
    common::assert_max_diff(&v_deq8_next, &w["v_deq8_next"], 1e-5)?;

    // 4. 4-bit momentum quantization parity
    let (m_deq4_init, m_int4_init, v_deq4_init, v_int4_init) =
        topic::quantize_4bit_moments(&w["m_prev"], &w["v_prev"], 16)?;
    common::assert_max_diff(&m_deq4_init, &w["m_deq4_init"], 1e-5)?;
    common::assert_max_diff(&m_int4_init, &w["m_int4_init"], 1e-5)?;
    common::assert_max_diff(&v_deq4_init, &w["v_deq4_init"], 1e-5)?;
    common::assert_max_diff(&v_int4_init, &w["v_int4_init"], 1e-5)?;

    // 5. 4-bit AdamW step parity
    let (theta_next_4bit, m_next4, v_next4) = topic::adamw_update_step(
        &w["theta"],
        &w["grad"],
        &w["m_deq4_init"],
        &w["v_deq4_init"],
        0.9,
        0.999,
        1e-3,
        1e-2,
        1,
    )?;
    common::assert_max_diff(&theta_next_4bit, &w["theta_next_4bit"], 1e-5)?;

    let (m_deq4_next, _, v_deq4_next, _) = topic::quantize_4bit_moments(&m_next4, &v_next4, 16)?;
    common::assert_max_diff(&m_deq4_next, &w["m_deq4_next"], 1e-5)?;
    common::assert_max_diff(&v_deq4_next, &w["v_deq4_next"], 1e-5)?;

    Ok(())
}
