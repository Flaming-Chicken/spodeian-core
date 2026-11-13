use candle_core::{Device, Result, safetensors::load};

use sml_candle::c07_domain_tasks::t11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/11_recommenders_dlrm_dcnv2_sasrec_and_tiger_semantic_ids.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. DLRM
    let dlrm = topic::dlrm_feature_interaction(&w["v_dense"], &w["v_sparse"])?;
    common::assert_max_diff(&dlrm, &w["dlrm_ref"], 1e-5)?;

    // 2. DCN-v2
    let dcn = topic::dcn_v2_cross_layer(&w["x0"], &w["xl"], &w["w_cross"], &w["b_cross"])?;
    common::assert_max_diff(&dcn, &w["dcn_ref"], 1e-5)?;

    // 3. SASRec
    let sasrec = topic::sasrec_causal_attention(&w["q"], &w["k"], &w["v"])?;
    common::assert_max_diff(&sasrec, &w["sasrec_ref"], 1e-5)?;

    // 4. RQ-VAE
    let (quantized, indices, next_res) =
        topic::rq_vae_quantize_step(&w["residual"], &w["codebook"])?;
    common::assert_max_diff(&quantized, &w["quantized_ref"], 1e-5)?;
    let diff_idx = (&indices.to_dtype(candle_core::DType::F32)?
        - &w["indices_ref"].to_dtype(candle_core::DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff_idx < 1e-5);
    common::assert_max_diff(&next_res, &w["next_res_ref"], 1e-5)?;

    Ok(())
}
