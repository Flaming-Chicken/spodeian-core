use candle_core::{Device, Result, safetensors::load};

use ai_candle::c14_quantisation::t16_trellis_and_e8_lattice_quantization_quip_sharp_qtip as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_16_trellis_and_e8_lattice_quantization_quip_sharp_qtip_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. E8 Lattice quantization parity
    let (w_e8_deq, lattice_points) = topic::quantize_e8_lattice(&w["w_vecs"], &w["e8_scale"])?;
    common::assert_max_diff(&w_e8_deq, &w["w_e8_deq"], 1e-5)?;
    common::assert_max_diff(&lattice_points, &w["lattice_points"], 1e-5)?;

    // 3. Trellis-Coded Quantization parity
    let tcq_recon =
        topic::trellis_coded_quantize_1d(&w["tcq_x"], &w["tcq_subsets"], &w["tcq_transitions"])?;
    common::assert_max_diff(&tcq_recon, &w["tcq_recon"], 1e-5)?;

    // 4. E8 Forward parity
    let (out_e8, w_full) = topic::e8_forward(
        &w["x_calib"],
        &w["w_orig"],
        &w["e8_scale"],
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&w_full, &w["w_deq_full"], 1e-5)?;
    common::assert_max_diff(&out_e8, &w["out_e8"], 1e-5)?;

    Ok(())
}
