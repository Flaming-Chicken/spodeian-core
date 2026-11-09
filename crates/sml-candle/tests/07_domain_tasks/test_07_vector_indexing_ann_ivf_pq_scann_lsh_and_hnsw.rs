use candle_core::{safetensors::load, Device, Result};

use ai_candle::c07_domain_tasks::t07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/07_vector_indexing_ann_ivf_pq_scann_lsh_and_hnsw.safetensors",
        ),
        &device,
    )?;

    // 0. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. PQ distance table & ADC distances
    let dist_table = topic::pq_distance_table(&w["query_subvectors"], &w["codebooks"])?;
    common::assert_max_diff(&dist_table, &w["dist_table"], 1e-5)?;

    let adc = topic::pq_asymmetric_distance(&w["dist_table"], &w["codes"])?;
    common::assert_max_diff(&adc, &w["adc_distances"], 1e-5)?;

    // 2. ScaNN Anisotropic Loss
    let scann = topic::scann_anisotropic_loss(&w["x"], &w["x_tilde"], 2.0, 1e-6)?;
    common::assert_max_diff(&scann, &w["scann_loss"], 1e-5)?;

    // 3. LSH Cosine Sim
    let lsh = topic::lsh_hash_hamming_distance(&w["q"], &w["d"], &w["hyperplanes"])?;
    common::assert_max_diff(&lsh, &w["lsh_sim"], 1e-5)?;

    Ok(())
}
