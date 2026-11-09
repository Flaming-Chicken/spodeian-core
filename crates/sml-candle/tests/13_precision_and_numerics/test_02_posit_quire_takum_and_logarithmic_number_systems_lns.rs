use candle_core::{safetensors::load, Device, Result};

use ai_candle::c13_precision_and_numerics::t02_posit_quire_takum_and_logarithmic_number_systems_lns as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_posit_quire_takum_and_logarithmic_number_systems_lns_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("13_precision_and_numerics/02_posit_quire_takum_and_logarithmic_number_systems_lns.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Posit Quantization
    let posit = topic::posit_tapered_quantize(&w["posit_in"], 8, 1)?;
    common::assert_max_diff(&posit, &w["posit_ref"], 1e-5)?;

    // 3. LNS Multiplication
    let lns_mul = topic::lns_multiplication(&w["lns_a"], &w["lns_b"], 1e-12)?;
    common::assert_max_diff(&lns_mul, &w["lns_mul_ref"], 1e-5)?;

    // 4. LNS Addition
    let lns_add = topic::lns_gaussian_addition(&w["add_a"], &w["add_b"], 1e-12)?;
    common::assert_max_diff(&lns_add, &w["lns_add_ref"], 1e-5)?;

    // 5. Quire Dot Product
    let quire = topic::quire_dot_product_sim(&w["quire_mat"], &w["quire_vec"])?;
    common::assert_max_diff(&quire, &w["quire_ref"], 1e-5)?;

    Ok(())
}
