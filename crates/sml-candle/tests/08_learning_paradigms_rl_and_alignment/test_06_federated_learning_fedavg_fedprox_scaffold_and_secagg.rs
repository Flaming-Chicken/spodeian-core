use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t06_federated_learning_fedavg_fedprox_scaffold_and_secagg as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_federated_learning_fedavg_fedprox_scaffold_and_secagg_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. FedAvg
    let agg = topic::fedavg_aggregate(&w["client_weights"], &w["client_counts"])?;
    common::assert_max_diff(&agg, &w["fedavg_aggregated_ref"], 1e-5)?;

    // 3. FedProx
    let w_prox =
        topic::fedprox_gradient_step(&w["w_local"], &w["w_global"], &w["grad_local"], 0.05, 0.1)?;
    common::assert_max_diff(&w_prox, &w["w_prox_ref"], 1e-5)?;

    // 4. SCAFFOLD
    let scaffold = topic::scaffold_correct_gradient(&w["grad_i"], &w["c_i"], &w["c_global"])?;
    common::assert_max_diff(&scaffold, &w["scaffold_corrected_ref"], 1e-5)?;

    // 5. SecAgg
    let secagg = topic::secagg_aggregate_sum(&w["masked_models"])?;
    common::assert_max_diff(&secagg, &w["secagg_unmasked_sum_ref"], 1e-5)?;

    Ok(())
}
