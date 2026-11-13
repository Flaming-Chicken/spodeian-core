use candle_core::{Device, Result, safetensors::load};

use ai_candle::c04_training_methods_and_optimisers::t08_constrained_proximal_frank_wolfe_admm_and_mirror_descent as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let param_pgd = weights.get("param_pgd").unwrap();
    let grad_pgd = weights.get("grad_pgd").unwrap();
    let golden_param_pgd = weights.get("golden_param_pgd").unwrap();

    let param_fw = weights.get("param_fw").unwrap();
    let grad_fw = weights.get("grad_fw").unwrap();
    let golden_param_fw = weights.get("golden_param_fw").unwrap();

    let param_ista = weights.get("param_ista").unwrap();
    let grad_ista = weights.get("grad_ista").unwrap();
    let golden_param_ista = weights.get("golden_param_ista").unwrap();

    let param_md = weights.get("param_md").unwrap();
    let grad_md = weights.get("grad_md").unwrap();
    let golden_param_md = weights.get("golden_param_md").unwrap();

    // 1. PGD
    let p_pgd = opt::step_pgd(param_pgd, grad_pgd, 0.1, 1.0, 1e-8)?;
    let diff_pgd = p_pgd
        .sub(golden_param_pgd)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_pgd < 1e-5, "PGD diff: {}", diff_pgd);

    // 2. Frank-Wolfe
    let p_fw = opt::step_frank_wolfe(param_fw, grad_fw, 1.0, 0.2)?;
    let diff_fw = p_fw
        .sub(golden_param_fw)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_fw < 1e-5, "Frank-Wolfe diff: {}", diff_fw);

    // 3. ISTA
    let p_ista = opt::step_ista(param_ista, grad_ista, 0.05, 0.1)?;
    let diff_ista = p_ista
        .sub(golden_param_ista)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_ista < 1e-5, "ISTA diff: {}", diff_ista);

    // 4. Mirror Descent
    let p_md = opt::step_mirror_descent(param_md, grad_md, 0.1, 1e-12)?;
    let diff_md = p_md
        .sub(golden_param_md)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_md < 1e-5, "Mirror Descent diff: {}", diff_md);

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
    let module = opt::ConstrainedOptimUnifiedModule::load_from_safetensors(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["pgd_err"] < 1e-5);
    assert!(errs["fw_err"] < 1e-5);
    assert!(errs["ista_err"] < 1e-5);
    assert!(errs["mirror_descent_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
