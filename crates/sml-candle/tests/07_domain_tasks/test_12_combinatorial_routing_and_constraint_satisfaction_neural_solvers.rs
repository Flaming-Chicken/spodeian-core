use candle_core::{Device, Result, safetensors::load};

use sml_candle::c07_domain_tasks::t12_combinatorial_routing_and_constraint_satisfaction_neural_solvers as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_combinatorial_routing_and_constraint_satisfaction_neural_solvers_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/12_combinatorial_routing_and_constraint_satisfaction_neural_solvers.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. TSP Tour Cost
    let cost = topic::tsp_tour_cost(&w["coords"], &w["tour"])?;
    common::assert_max_diff(&cost, &w["tsp_cost_ref"], 1e-5)?;

    // 2. Pointer Network
    let probs = topic::pointer_network_step(
        &w["node_embeds"],
        &w["decoder_state"],
        &w["visited_mask"],
        &w["w1"],
        &w["w2"],
        &w["v"],
    )?;
    common::assert_max_diff(&probs, &w["ptr_probs_ref"], 1e-5)?;

    // 3. GNN Edge Heatmap
    let heatmap =
        topic::gnn_edge_heatmap(&w["h_nodes"], &w["edge_dists"], &w["w_edge"], &w["b_edge"])?;
    common::assert_max_diff(&heatmap, &w["heatmap_ref"], 1e-5)?;

    // 4. Relaxed Max-Cut Cost
    let cut = topic::relaxed_maxcut_cost(&w["z_spins"], &w["w_adj"])?;
    common::assert_max_diff(&cut, &w["maxcut_cost_ref"], 1e-5)?;

    Ok(())
}
