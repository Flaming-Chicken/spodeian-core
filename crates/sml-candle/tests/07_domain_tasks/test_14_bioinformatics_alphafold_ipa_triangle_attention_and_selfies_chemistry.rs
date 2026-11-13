use candle_core::{Device, Result, safetensors::load};

use ai_candle::c07_domain_tasks::t14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Triangle Multiplicative Update
    let tri_out = topic::triangle_multiplicative_update(
        &w["z"], &w["w_a"], &w["b_a"], &w["w_b"], &w["b_b"], &w["w_g"], &w["b_g"], &w["w_o"],
        &w["b_o"],
    )?;
    common::assert_max_diff(&tri_out, &w["tri_out_ref"], 1e-5)?;

    // 2. FAPE Loss
    let fape = topic::fape_loss(
        &w["x"], &w["y"], &w["r_x"], &w["t_x"], &w["r_y"], &w["t_y"], 10.0,
    )?;
    common::assert_max_diff(&fape, &w["fape_loss_ref"], 1e-5)?;

    // 3. IPA Distance Penalty
    let ipa = topic::ipa_point_distance_penalty(&w["q_pts"], &w["k_pts"], &w["gamma"])?;
    common::assert_max_diff(&ipa, &w["ipa_penalty_ref"], 1e-5)?;

    // 4. Molecular Valence Loss
    let valence = topic::molecular_valence_loss(&w["bond_logits"], &w["max_valence"])?;
    common::assert_max_diff(&valence, &w["valence_loss_ref"], 1e-5)?;

    Ok(())
}
