use candle_core::{Device, Result, safetensors::load};

use sml_candle::c09_causal_ai_and_evolutionary_search::t05_quality_diversity_map_elites_and_differentiable_qd as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_quality_diversity_map_elites_and_differentiable_qd_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. MAP-Elites grid coordinates & cell indices
    let (grid_idx, cell_idx) = topic::map_elites_grid_indices(
        &w["descriptors"],
        &w["b_min"],
        &w["b_max"],
        &w["grid_size"],
    )?;
    common::assert_max_diff(&grid_idx, &w["grid_indices_ref"], 1e-5)?;
    common::assert_max_diff(&cell_idx, &w["cell_indices_ref"], 1e-5)?;

    // 3. MAP-Elites archive update
    let updated_arch = topic::map_elites_archive_update(
        &w["archive_fit"],
        &w["cand_fitness"],
        &w["cell_indices_ref"],
    )?;
    common::assert_max_diff(&updated_arch, &w["archive_fit_updated_ref"], 1e-5)?;

    // 4. DQD gradient arborescence step
    let step = topic::dqd_gradient_arborescence_step(&w["grad_f"], &w["jac_b"], &w["dqd_coeffs"])?;
    common::assert_max_diff(&step, &w["dqd_step_ref"], 1e-5)?;

    // 5. DQD orthogonalized behavioral gradients
    let ortho = topic::dqd_orthogonalize_behavior_gradients(&w["grad_f"], &w["jac_b"], 1e-8)?;
    common::assert_max_diff(&ortho, &w["dqd_ortho_jac_ref"], 1e-5)?;

    Ok(())
}
