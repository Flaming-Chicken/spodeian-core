use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t40_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_40_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers.safetensors",
        ),
        &device,
    )?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let (t_norm, t_conorm, imp) = topic::lukasiewicz_logic(&w["logic_a"], &w["logic_b"])?;
    common::assert_max_diff(&t_norm, &w["t_norm"], 1e-5)?;
    common::assert_max_diff(&t_conorm, &w["t_conorm"], 1e-5)?;
    common::assert_max_diff(&imp, &w["implication"], 1e-5)?;

    let (bound, bundled, sim) = topic::vsa_bind_and_bundle(
        &w["vsa_key"],
        &w["vsa_val"],
        &w["vsa_other"],
        &w["vsa_codebook"],
    )?;
    common::assert_max_diff(&bound, &w["vsa_bound"], 1e-5)?;
    common::assert_max_diff(&bundled, &w["vsa_bundled"], 1e-5)?;
    common::assert_max_diff(&sim, &w["vsa_sim"], 1e-5)?;

    let p = topic::sinkhorn_knopp_solver(&w["sinkhorn_logits"], 5, 0.5)?;
    common::assert_max_diff(&p, &w["sinkhorn_p"], 1e-5)?;

    Ok(())
}
