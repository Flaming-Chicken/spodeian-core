use candle_core::{Device, Result, safetensors::load};

use ai_candle::c07_domain_tasks::t06_nlp_dense_retrieval_colbert_ubinary_and_rag as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_nlp_dense_retrieval_colbert_ubinary_and_rag_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/06_nlp_dense_retrieval_colbert_ubinary_and_rag.safetensors",
        ),
        &device,
    )?;

    // 0. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. ColBERT MaxSim
    let maxsim = topic::colbert_maxsim(&w["query_embeds"], &w["doc_embeds"])?;
    common::assert_max_diff(&maxsim, &w["maxsim_ref"], 1e-5)?;

    // 2. UBinary
    let ub = topic::ubinary_quantize_and_dot(&w["q_float"], &w["d_float"])?;
    common::assert_max_diff(&ub, &w["ubinary_ref"], 1e-5)?;

    // 3. Reciprocal Rank Fusion
    let rrf = topic::reciprocal_rank_fusion(&w["ranks1"], &w["ranks2"], 60.0)?;
    common::assert_max_diff(&rrf, &w["rrf_ref"], 1e-5)?;

    // 4. Dense InfoNCE Loss
    let loss = topic::dense_infonce_loss(&w["q_vecs"], &w["p_vecs"], 0.05)?;
    common::assert_max_diff(&loss, &w["infonce_ref"], 1e-5)?;

    Ok(())
}
