use candle_core::{safetensors::load, Device, Result};

use ai_candle::c13_precision_and_numerics::t01_floating_point_anatomy_subnormal_traps_and_stochastic_logic as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_floating_point_anatomy_subnormal_traps_and_stochastic_logic_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. FTZ / DAZ simulation
    let ftz = topic::ftz_daz_simulate(&w["raw_vals"], 1e-4)?;
    common::assert_max_diff(&ftz, &w["ftz_ref"], 1e-5)?;

    // 3. BF16 RTNE conversion
    let bf16 = topic::fp32_to_bf16_rtne(&w["bf16_in"])?;
    common::assert_max_diff(&bf16, &w["bf16_ref"], 1e-5)?;

    // 4. Stochastic Unipolar multiplication
    let unipolar = topic::stochastic_unipolar_mul(&w["p1"], &w["p2"], &w["u1"], &w["u2"])?;
    common::assert_max_diff(&unipolar, &w["unipolar_ref"], 1e-5)?;

    // 5. Stochastic Bipolar multiplication
    let bipolar = topic::stochastic_bipolar_mul(&w["x1"], &w["x2"], &w["u1"], &w["u2"])?;
    common::assert_max_diff(&bipolar, &w["bipolar_ref"], 1e-5)?;

    Ok(())
}
