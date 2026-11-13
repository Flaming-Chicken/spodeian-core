use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t05_kolmogorov_arnold_networks_kan as kan;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_kolmogorov_arnold_networks_kan_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();

    // 1. ChebyKAN verification
    let cheby_c = weights.get("cheby_c").unwrap();
    let expected_cheby = weights.get("chebykan_output").unwrap();

    let cheby_out = kan::chebykan_forward(x, cheby_c, 4)?;
    let diff_cheby = cheby_out
        .sub(expected_cheby)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_cheby < 1e-5, "ChebyKAN parity error: {}", diff_cheby);

    // 2. FastKAN verification
    let base_w = weights.get("fastkan_base_w").unwrap();
    let rbf_w = weights.get("fastkan_rbf_w").unwrap();
    let grid = weights.get("fastkan_grid").unwrap();
    let denom_t = weights.get("fastkan_denominator").unwrap();
    let denominator = denom_t.to_scalar::<f32>()? as f64;
    let expected_fastkan = weights.get("fastkan_output").unwrap();

    let fastkan_out = kan::fastkan_forward(x, base_w, rbf_w, grid, denominator)?;
    let diff_fastkan = fastkan_out
        .sub(expected_fastkan)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_fastkan < 1e-5,
        "FastKAN parity error: {}",
        diff_fastkan
    );

    Ok(())
}
