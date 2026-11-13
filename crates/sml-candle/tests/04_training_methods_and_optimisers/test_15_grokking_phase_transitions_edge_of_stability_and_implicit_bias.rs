use candle_core::{Device, Result, safetensors::load};

use sml_candle::c04_training_methods_and_optimisers::t15_grokking_phase_transitions_edge_of_stability_and_implicit_bias as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_15_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let g_raw = weights.get("g_raw").unwrap();
    let mu_prev = weights.get("mu_prev").unwrap();
    let golden_mu = weights.get("golden_mu").unwrap();
    let golden_g_grokfast = weights.get("golden_g_grokfast").unwrap();

    let x_eos = weights.get("x_eos").unwrap();
    let h_eos = weights.get("h_eos").unwrap();
    let gamma_eos = weights.get("gamma_eos").unwrap();
    let golden_x_eos = weights.get("golden_x_eos").unwrap();

    let w_sep = weights.get("w_sep").unwrap();
    let x_sep = weights.get("x_sep").unwrap();
    let y_sep = weights.get("y_sep").unwrap();
    let golden_w_sep = weights.get("golden_w_sep").unwrap();

    let anchor_w = weights.get("anchor_w").unwrap();
    let anchor_b = weights.get("anchor_b").unwrap();
    let anchor_x = weights.get("anchor_x").unwrap();
    let golden_anchor_out = weights.get("golden_anchor_out").unwrap();

    // 1. Grokfast Invariant
    let (g_gf, mu_next) = opt::step_grokfast(g_raw, mu_prev, 0.8, 2.0)?;
    let diff_mu = mu_next
        .sub(golden_mu)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_gf = g_gf
        .sub(golden_g_grokfast)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_mu < 1e-5, "Grokfast mu diff: {}", diff_mu);
    assert!(diff_gf < 1e-5, "Grokfast filtered g diff: {}", diff_gf);

    // 2. EOS Invariant
    let x_next = opt::step_edge_of_stability(x_eos, h_eos, gamma_eos, 0.1)?;
    let diff_eos = x_next
        .sub(golden_x_eos)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_eos < 1e-5, "EOS diff: {}", diff_eos);

    // 3. Implicit Bias Invariant
    let w_next = opt::step_implicit_bias(w_sep, x_sep, y_sep, 0.2)?;
    let diff_w = w_next
        .sub(golden_w_sep)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_w < 1e-5, "Implicit bias diff: {}", diff_w);

    // 4. Linear Anchor Invariant
    let anchor = opt::LinearAnchor::new(anchor_w.clone(), anchor_b.clone());
    let anchor_out = anchor.forward(anchor_x)?;
    let diff_anchor = anchor_out
        .sub(golden_anchor_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "Anchor diff: {}", diff_anchor);

    Ok(())
}
