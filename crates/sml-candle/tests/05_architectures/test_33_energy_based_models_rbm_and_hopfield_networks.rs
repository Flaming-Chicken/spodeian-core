use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t33_energy_based_models_rbm_and_hopfield_networks as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_33_energy_based_models_rbm_and_hopfield_networks_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.safetensors",
        ),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // RBM Free energy
    let free_energy = topic::rbm_free_energy(&w["v"], &w["w_rbm"], &w["a_rbm"], &w["b_rbm"])?;
    common::assert_max_diff(&free_energy, &w["free_energy"], 1e-5)?;

    // Langevin dynamics step
    let x_langevin =
        topic::langevin_dynamics_step(&w["x_ebm"], &w["grad_energy"], 0.01, &w["noise"])?;
    common::assert_max_diff(&x_langevin, &w["x_langevin"], 1e-5)?;

    // Modern Hopfield energy
    let hopfield_energy = topic::modern_hopfield_energy(&w["x_ebm"], &w["memories"], 2.0)?;
    common::assert_max_diff(&hopfield_energy, &w["hopfield_energy"], 1e-5)?;

    // Modern Hopfield retrieve
    let retrieved = topic::modern_hopfield_retrieve(&w["x_ebm"], &w["memories"], 2.0)?;
    common::assert_max_diff(&retrieved, &w["retrieved"], 1e-5)?;

    Ok(())
}
