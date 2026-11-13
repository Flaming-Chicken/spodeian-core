use candle_core::{Device, Result, safetensors::load};

use ai_candle::c04_training_methods_and_optimisers::t09_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let param = weights.get("param").unwrap();
    let g1 = weights.get("g1").unwrap();
    let g2 = weights.get("g2").unwrap();
    let losses = weights.get("L").unwrap();
    let z_star = weights.get("z_star").unwrap();
    let weights_tch = weights.get("weights_tch").unwrap();
    let weights_lin = weights.get("weights_lin").unwrap();

    let golden_alpha_mgda = weights.get("golden_alpha_mgda").unwrap().to_vec1::<f32>()?[0] as f64;
    let golden_g_mgda = weights.get("golden_g_mgda").unwrap();
    let golden_param_mgda = weights.get("golden_param_mgda").unwrap();
    let golden_g_pc = weights.get("golden_g_pc").unwrap();
    let golden_param_pc = weights.get("golden_param_pc").unwrap();
    let golden_g_tch = weights.get("golden_g_tch").unwrap();
    let golden_param_tch = weights.get("golden_param_tch").unwrap();
    let golden_g_lin = weights.get("golden_g_lin").unwrap();
    let golden_param_lin = weights.get("golden_param_lin").unwrap();

    // 1. MGDA
    let (p_mgda, g_mgda, alpha) = opt::step_mgda_2task(param, g1, g2, 0.01, 1e-12)?;
    assert!((alpha - golden_alpha_mgda).abs() < 1e-5);
    let diff_g_mgda = g_mgda
        .sub(golden_g_mgda)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_p_mgda = p_mgda
        .sub(golden_param_mgda)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_g_mgda < 1e-5, "MGDA g diff: {}", diff_g_mgda);
    assert!(diff_p_mgda < 1e-5, "MGDA param diff: {}", diff_p_mgda);

    // 2. PCGrad
    let (p_pc, g_pc) = opt::step_pcgrad_2task(param, g1, g2, 0.01, 1e-12)?;
    let diff_g_pc = g_pc
        .sub(golden_g_pc)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_p_pc = p_pc
        .sub(golden_param_pc)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_g_pc < 1e-5, "PCGrad g diff: {}", diff_g_pc);
    assert!(diff_p_pc < 1e-5, "PCGrad param diff: {}", diff_p_pc);

    // 3. Tchebycheff
    let (p_tch, g_tch) =
        opt::step_tchebycheff(param, g1, g2, losses, z_star, weights_tch, 0.05, 0.01)?;
    let diff_g_tch = g_tch
        .sub(golden_g_tch)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_p_tch = p_tch
        .sub(golden_param_tch)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_g_tch < 1e-5, "Tchebycheff g diff: {}", diff_g_tch);
    assert!(diff_p_tch < 1e-5, "Tchebycheff param diff: {}", diff_p_tch);

    // 4. Linear Scalarization
    let (p_lin, g_lin) = opt::step_linear_scalarization(param, g1, g2, weights_lin, 0.01)?;
    let diff_g_lin = g_lin
        .sub(golden_g_lin)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_p_lin = p_lin
        .sub(golden_param_lin)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_g_lin < 1e-5, "Linear g diff: {}", diff_g_lin);
    assert!(diff_p_lin < 1e-5, "Linear param diff: {}", diff_p_lin);

    // 5. Linear Anchor
    let anchor_w = weights.get("anchor_w").unwrap();
    let anchor_b = weights.get("anchor_b").unwrap();
    let anchor_x = weights.get("anchor_x").unwrap();
    let anchor = opt::LinearAnchor::new(anchor_w.clone(), anchor_b.clone());
    let anchor_out = anchor.forward(anchor_x)?;
    let expected_anchor = weights.get("golden_anchor_out").unwrap();
    let diff_anchor = anchor_out
        .sub(expected_anchor)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "anchor diff: {}", diff_anchor);

    // Unified module verification
    let module = opt::MultiObjectiveUnifiedModule::load_from_safetensors(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["mgda_err"] < 1e-5);
    assert!(errs["pcgrad_err"] < 1e-5);
    assert!(errs["tchebycheff_err"] < 1e-5);
    assert!(errs["linear_scalarization_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
