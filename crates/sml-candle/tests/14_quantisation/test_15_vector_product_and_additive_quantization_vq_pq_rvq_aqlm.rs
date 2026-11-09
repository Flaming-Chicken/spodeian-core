use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t15_vector_product_and_additive_quantization_vq_pq_rvq_aqlm as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_15_vector_product_and_additive_quantization_vq_pq_rvq_aqlm_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("14_quantisation/04_vector_lattice_qat_and_lowbit/01_vector_product_and_additive_quantization_vq_pq_rvq_aqlm.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. VQ parity
    let (vq_recon, vq_idx) = topic::vector_quantize(&w["w_vecs"], &w["vq_codebook"])?;
    common::assert_max_diff(&vq_recon, &w["vq_recon"], 1e-5)?;
    common::assert_max_diff(&vq_idx, &w["vq_indices"], 1e-5)?;

    // 3. PQ parity
    let (pq_recon, pq_idx) = topic::product_quantize(&w["w_vecs"], &w["pq_codebooks"])?;
    common::assert_max_diff(&pq_recon, &w["pq_recon"], 1e-5)?;
    common::assert_max_diff(&pq_idx, &w["pq_indices"], 1e-5)?;

    // 4. RVQ parity
    let (rvq_recon, rvq_idx) = topic::residual_vector_quantize(&w["w_vecs"], &w["rvq_codebooks"])?;
    common::assert_max_diff(&rvq_recon, &w["rvq_recon"], 1e-5)?;
    common::assert_max_diff(&rvq_idx, &w["rvq_indices"], 1e-5)?;

    // 5. AQLM forward parity
    let (out_aqlm, w_aqlm_deq) = topic::aqlm_forward(
        &w["x_calib"],
        &w["aqlm_indices"],
        &w["aqlm_codebooks"],
        16,
        8,
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&w_aqlm_deq, &w["w_aqlm_deq"], 1e-5)?;
    common::assert_max_diff(&out_aqlm, &w["out_aqlm"], 1e-5)?;

    Ok(())
}
