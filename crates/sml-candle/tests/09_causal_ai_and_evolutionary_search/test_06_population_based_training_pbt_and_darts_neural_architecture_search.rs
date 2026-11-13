use candle_core::{Device, Result, safetensors::load};

use sml_candle::c09_causal_ai_and_evolutionary_search::t06_population_based_training_pbt_and_darts_neural_architecture_search as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_population_based_training_pbt_and_darts_neural_architecture_search_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "09_causal_ai_and_evolutionary_search/06_population_based_training_pbt_and_darts_neural_architecture_search.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. PBT exploit & explore step
    let (w_up, h_up) = topic::pbt_exploit_explore_step(
        &w["pbt_scores"],
        &w["pbt_weights"],
        &w["pbt_hyperparams"],
        &w["pbt_perturb"],
    )?;
    common::assert_max_diff(&w_up, &w["pbt_weights_updated_ref"], 1e-5)?;
    common::assert_max_diff(&h_up, &w["pbt_hyperparams_updated_ref"], 1e-5)?;

    // 3. DARTS continuous mixed-op forward pass
    let (mixed_out, softmax_w) =
        topic::darts_mixed_op_forward(&w["darts_alpha"], &w["darts_candidate_ops"])?;
    common::assert_max_diff(&mixed_out, &w["darts_mixed_out_ref"], 1e-5)?;
    common::assert_max_diff(&softmax_w, &w["darts_softmax_weights_ref"], 1e-5)?;

    // 4. DARTS discrete architecture decoding
    let (ops, disc_out) =
        topic::darts_discrete_decode(&w["darts_alpha"], &w["darts_candidate_ops"])?;
    common::assert_max_diff(&ops, &w["darts_selected_ops_ref"], 1e-5)?;
    common::assert_max_diff(&disc_out, &w["darts_discrete_out_ref"], 1e-5)?;

    // 5. DARTS architecture entropy
    let ent = topic::darts_architecture_entropy(&softmax_w, 1e-8)?;
    common::assert_max_diff(&ent, &w["darts_entropy_ref"], 1e-5)?;

    Ok(())
}
