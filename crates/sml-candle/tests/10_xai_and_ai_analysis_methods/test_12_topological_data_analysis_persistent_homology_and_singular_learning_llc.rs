use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t12_topological_data_analysis_persistent_homology_and_singular_learning_llc as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_topological_data_analysis_persistent_homology_and_singular_learning_llc_parity(
) -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Persistence lifespans
    let lifespans = topic::persistence_diagram_lifespans(&w["birth_death_pairs"])?;
    common::assert_max_diff(&lifespans, &w["lifespans_ref"], 1e-5)?;

    // 3. Persistence landscape level 1
    let landscape =
        topic::persistence_landscape_level1(&w["birth_death_pairs"], &w["eval_points"])?;
    common::assert_max_diff(&landscape, &w["landscape_ref"], 1e-5)?;

    // 4. Singular Learning Theory LLC
    let llc =
        topic::singular_learning_llc_energy(&w["sample_losses"], &w["loss_star"], 500.0, 1.0)?;
    common::assert_max_diff(&llc, &w["llc_ref"], 1e-5)?;

    // 5. Betti-0 filtration counts
    let betti0 =
        topic::betti_0_filtration_counts(&w["pairwise_distances"], &w["epsilon_thresholds"])?;
    common::assert_max_diff(&betti0, &w["betti0_ref"], 1e-5)?;

    Ok(())
}
