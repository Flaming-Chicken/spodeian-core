use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t38_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_38_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/06_graphs_geometric_and_inr/06_hypergraph_neural_networks_allset_combinatorial_complexes_and_exphormer.safetensors",
        ),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Hypergraph convolution
    let h_hg = topic::hypergraph_conv(&w["x"], &w["inc"], &w["w_hg"], &w["b_hg"])?;
    common::assert_max_diff(&h_hg, &w["h_hg"], 1e-5)?;

    // AllSet step
    let h_allset = topic::allset_multiset_step(
        &w["x"], &w["inc"], &w["w_v"], &w["b_v"], &w["w_e"], &w["b_e"],
    )?;
    common::assert_max_diff(&h_allset, &w["h_allset"], 1e-5)?;

    // Exphormer spectral mixing
    let h_exphormer =
        topic::exphormer_spectral_mixing(&w["x"], &w["exp_adj"], &w["w_exp"], &w["b_exp"])?;
    common::assert_max_diff(&h_exphormer, &w["h_exphormer"], 1e-5)?;

    Ok(())
}
