use candle_core::{Device, Result, safetensors::load};

use sml_candle::c13_precision_and_numerics::t06_stochastic_rounding_and_master_weight_free_bf16_training as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_stochastic_rounding_and_master_weight_free_bf16_training_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Stochastic Rounding BF16
    let sr_out = topic::stochastic_rounding_bf16(&w["x_float"], &w["noise"])?;
    common::assert_max_diff(&sr_out, &w["sr_out_ref"], 1e-5)?;

    let rtne_out = topic::deterministic_rtne_bf16(&w["x_float"])?;
    common::assert_max_diff(&rtne_out, &w["rtne_out_ref"], 1e-5)?;

    // 3. Master-weight-free SGD
    let sgd_sr =
        topic::master_weight_free_sgd_step(&w["w_init"], &w["grad_tiny"], 0.01, &w["noise"], true)?;
    common::assert_max_diff(&sgd_sr, &w["sgd_sr_ref"], 1e-5)?;

    let sgd_rtne = topic::master_weight_free_sgd_step(
        &w["w_init"],
        &w["grad_tiny"],
        0.01,
        &w["noise"],
        false,
    )?;
    common::assert_max_diff(&sgd_rtne, &w["sgd_rtne_ref"], 1e-5)?;

    // 4. Kahan accumulation
    let (k_accum, k_comp) = topic::kahan_accumulate_step(&w["k_accum"], &w["k_val"], &w["k_comp"])?;
    common::assert_max_diff(&k_accum, &w["k_accum_ref"], 1e-5)?;
    common::assert_max_diff(&k_comp, &w["k_comp_ref"], 1e-5)?;

    Ok(())
}
