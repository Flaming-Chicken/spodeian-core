use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t20_qlora_loftq_and_qa_lora_quantized_peft as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_20_qlora_loftq_and_qa_lora_quantized_peft_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("14_quantisation/04_vector_lattice_qat_and_lowbit/06_qlora_loftq_and_qa_lora_quantized_peft.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. NF4 quantization parity
    let (w_nf4_deq, nf4_indices, c1) = topic::nf4_quantize(&w["w_orig"], 8)?;
    common::assert_max_diff(&w_nf4_deq, &w["w_nf4_deq"], 1e-5)?;
    common::assert_max_diff(&nf4_indices, &w["nf4_indices"], 1e-5)?;
    common::assert_max_diff(&c1, &w["c1"], 1e-5)?;

    // 3. Double Quantization parity
    let (c1_deq, q2, c2) = topic::double_quantize(&w["c1"], 8)?;
    common::assert_max_diff(&c1_deq, &w["c1_deq"], 1e-5)?;
    common::assert_max_diff(&q2, &w["q2"], 1e-5)?;
    common::assert_max_diff(&c2, &w["c2"], 1e-5)?;

    // 4. QLoRA forward pass parity
    let scaling = 2.0 / 4.0;
    let out_qlora = topic::qlora_forward(
        &w["x_raw"],
        &w["w_dq_deq"],
        &w["lora_a"],
        &w["lora_b"],
        scaling,
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&out_qlora, &w["out_qlora"], 1e-5)?;

    // 5. LoftQ forward pass parity
    let out_loftq = topic::qlora_forward(
        &w["x_raw"],
        &w["loftq_q_deq"],
        &w["loftq_a"],
        &w["loftq_b"],
        1.0,
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&out_loftq, &w["out_loftq"], 1e-5)?;

    // 6. QA-LoRA forward pass parity
    let (out_qa_lora, a_qa) = topic::qa_lora_forward(
        &w["x_raw"],
        &w["w_nf4_deq"],
        &w["lora_a"],
        &w["lora_b"],
        &w["group_scales"],
        scaling,
        4,
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&out_qa_lora, &w["out_qa_lora"], 1e-5)?;
    common::assert_max_diff(&a_qa, &w["a_qa"], 1e-5)?;

    Ok(())
}
