use candle_core::{safetensors::load, Device, Result};

use ai_candle::c04_training_methods_and_optimisers::t05_second_order_and_curvature_optimisers as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let theta = weights.get("theta").unwrap();
    let grad = weights.get("grad").unwrap();
    let momentum = weights.get("momentum").unwrap();
    let hess_diag = weights.get("hess_diag").unwrap();
    let hess_diag_est = weights.get("hess_diag_est").unwrap();
    let ada_v = weights.get("ada_v").unwrap();
    let ada_d = weights.get("ada_d").unwrap();
    let lbfgs_s = weights.get("lbfgs_s").unwrap();
    let lbfgs_y = weights.get("lbfgs_y").unwrap();
    let lm_jacobian = weights.get("lm_jacobian").unwrap();
    let lm_residual = weights.get("lm_residual").unwrap();

    // 1. L-BFGS
    let lbfgs_theta = opt::step_lbfgs(theta, lbfgs_s, lbfgs_y, grad, 0.1)?;
    let expected_lbfgs_theta = weights.get("golden_lbfgs_theta").unwrap();
    let diff_lbfgs = lbfgs_theta
        .sub(expected_lbfgs_theta)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_lbfgs < 1e-5, "lbfgs diff: {}", diff_lbfgs);

    // 2. Sophia-G
    let (sophia_theta, sophia_m, sophia_h) = opt::step_sophia(
        theta,
        grad,
        momentum,
        hess_diag,
        hess_diag_est,
        0.01,
        0.96,
        0.99,
        0.01,
        0.04,
        1e-6,
    )?;
    let expected_sophia_theta = weights.get("golden_sophia_theta").unwrap();
    let expected_sophia_m = weights.get("golden_sophia_m").unwrap();
    let expected_sophia_h = weights.get("golden_sophia_h").unwrap();
    let diff_sophia = sophia_theta
        .sub(expected_sophia_theta)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_sophia_m = sophia_m
        .sub(expected_sophia_m)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_sophia_h = sophia_h
        .sub(expected_sophia_h)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_sophia < 1e-5, "sophia diff: {}", diff_sophia);
    assert!(diff_sophia_m < 1e-5, "sophia_m diff: {}", diff_sophia_m);
    assert!(diff_sophia_h < 1e-5, "sophia_h diff: {}", diff_sophia_h);

    // 3. AdaHessian
    let (ada_theta, ada_m, ada_v_out) = opt::step_adahessian(
        theta, grad, momentum, ada_v, ada_d, 0.01, 0.9, 0.999, 0.01, 1, 1e-4,
    )?;
    let expected_ada_theta = weights.get("golden_adahessian_theta").unwrap();
    let expected_ada_m = weights.get("golden_adahessian_m").unwrap();
    let expected_ada_v = weights.get("golden_adahessian_v").unwrap();
    let diff_ada = ada_theta
        .sub(expected_ada_theta)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_ada_m = ada_m
        .sub(expected_ada_m)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_ada_v = ada_v_out
        .sub(expected_ada_v)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_ada < 1e-5, "adahessian diff: {}", diff_ada);
    assert!(diff_ada_m < 1e-5, "adahessian_m diff: {}", diff_ada_m);
    assert!(diff_ada_v < 1e-5, "adahessian_v diff: {}", diff_ada_v);

    // 4. Levenberg-Marquardt
    let lm_theta = opt::step_levenberg_marquardt(theta, lm_jacobian, lm_residual, 0.1)?;
    let expected_lm_theta = weights.get("golden_lm_theta").unwrap();
    let diff_lm = lm_theta
        .sub(expected_lm_theta)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_lm < 1e-5, "lm diff: {}", diff_lm);

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
    let module =
        opt::SecondOrderCurvatureUnifiedModule::load_from_safetensors(&weights_path, &device)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["lbfgs_err"] < 1e-5);
    assert!(errs["sophia_err"] < 1e-5);
    assert!(errs["adahessian_err"] < 1e-5);
    assert!(errs["lm_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
