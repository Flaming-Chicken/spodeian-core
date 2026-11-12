use candle_core::{safetensors::load, Device, Result, Tensor};

use ai_candle::c14_quantisation::t25_hawq_mixed_precision_bit_allocation_and_numeric_debugging as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_25_hawq_mixed_precision_bit_allocation_and_numeric_debugging_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/05_kv_attention_optim_and_benchmarks/05_hawq_mixed_precision_bit_allocation_and_numeric_debugging.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let weights = vec![
        w["w0"].clone(),
        w["w1"].clone(),
        w["w2"].clone(),
        w["w3"].clone(),
    ];
    let traces_vec = w["traces"].to_vec1::<f32>()?;
    let traces: Vec<f64> = traces_vec.into_iter().map(|x| x as f64).collect();

    // Bit allocation parity at target 3.5
    let alloc_3_5 = topic::greedy_hawq_bit_allocation(&weights, &traces, 3.5, &[2, 4, 8])?;
    let t_alloc_3_5 = Tensor::from_vec(
        alloc_3_5.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        (4,),
        &device,
    )?;
    common::assert_max_diff(&t_alloc_3_5, &w["alloc_target_3_5"], 1e-5)?;

    // Bit allocation parity at target 5.0
    let alloc_5_0 = topic::greedy_hawq_bit_allocation(&weights, &traces, 5.0, &[2, 4, 8])?;
    let t_alloc_5_0 = Tensor::from_vec(
        alloc_5_0.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        (4,),
        &device,
    )?;
    common::assert_max_diff(&t_alloc_5_0, &w["alloc_target_5_0"], 1e-5)?;

    // Layerwise SQNR diagnostic
    let (sqnrs, q_weights, fp_acts, q_acts) =
        topic::layerwise_sqnr_diagnostic(&w["x"], &weights, &alloc_5_0)?;

    for i in 0..4 {
        common::assert_max_diff(&q_weights[i], &w[&format!("w{i}_q")], 1e-5)?;
        common::assert_max_diff(&fp_acts[i], &w[&format!("a{i}")], 1e-5)?;
        common::assert_max_diff(&q_acts[i], &w[&format!("a{i}_q")], 1e-5)?;
    }

    common::assert_max_diff(&sqnrs, &w["sqnrs"], 1e-4)?;

    Ok(())
}
