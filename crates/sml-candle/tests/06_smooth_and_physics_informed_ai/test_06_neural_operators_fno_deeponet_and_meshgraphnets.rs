use candle_core::{safetensors::load, Device, Result};

use ai_candle::c06_smooth_and_physics_informed_ai::t06_neural_operators_fno_deeponet_and_meshgraphnets as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_neural_operators_fno_deeponet_and_meshgraphnets_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // FNO-1D
    let (v0, k_v0, v1, fno_output) = topic::fno_forward(
        &w["fno_input"],
        &w["w_p"],
        &w["b_p"],
        &w["r_real"],
        &w["r_imag"],
        &w["w_bypass"],
        &w["w_q"],
        &w["b_q"],
    )?;
    common::assert_max_diff(&v0, &w["v0"], 1e-5)?;
    common::assert_max_diff(&k_v0, &w["k_v0"], 1e-5)?;
    common::assert_max_diff(&v1, &w["v1"], 1e-5)?;
    common::assert_max_diff(&fno_output, &w["fno_output"], 1e-5)?;

    // DeepONet
    let (branch_feat, trunk_feat, deeponet_output) = topic::deeponet_forward(
        &w["u_sensor"],
        &w["y_query"],
        &w["w_b"],
        &w["b_b"],
        &w["w_t"],
        &w["b_t"],
        &w["b_out0"],
    )?;
    common::assert_max_diff(&branch_feat, &w["branch_feat"], 1e-5)?;
    common::assert_max_diff(&trunk_feat, &w["trunk_feat"], 1e-5)?;
    common::assert_max_diff(&deeponet_output, &w["deeponet_output"], 1e-5)?;

    // MeshGraphNets
    let src = [0usize, 1, 2, 3];
    let dst = [1usize, 2, 3, 0];
    let mgn_output = topic::meshgraphnet_step(
        &w["v_nodes"],
        &src,
        &dst,
        &w["e_feat"],
        &w["w_e"],
        &w["b_e"],
        &w["w_v"],
        &w["b_v"],
    )?;
    common::assert_max_diff(&mgn_output, &w["mgn_output"], 1e-5)?;

    Ok(())
}
