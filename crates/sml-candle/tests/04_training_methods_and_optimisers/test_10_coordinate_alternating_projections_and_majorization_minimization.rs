use candle_core::{safetensors::load, Device, Result};

use ai_candle::c04_training_methods_and_optimisers::t10_coordinate_alternating_projections_and_majorization_minimization as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let x0 = weights.get("x0").unwrap();
    let normal = weights.get("normal").unwrap();
    let offset = weights.get("offset").unwrap().to_vec1::<f32>()?[0] as f64;

    let param_bcd = weights.get("param_bcd").unwrap();
    let b_bcd = weights.get("b_bcd").unwrap();
    let a_diag = weights.get("A_diag").unwrap();

    let param_irls = weights.get("param_irls").unwrap();
    let target_irls = weights.get("target_irls").unwrap();

    let param_cccp = weights.get("param_cccp").unwrap();

    let golden_dykstra_x = weights.get("golden_dykstra_x").unwrap();
    let golden_dykstra_y = weights.get("golden_dykstra_y").unwrap();
    let golden_dykstra_p = weights.get("golden_dykstra_p").unwrap();
    let golden_dykstra_q = weights.get("golden_dykstra_q").unwrap();

    let golden_param_bcd = weights.get("golden_param_bcd").unwrap();
    let golden_param_irls = weights.get("golden_param_irls").unwrap();
    let golden_param_cccp = weights.get("golden_param_cccp").unwrap();

    // 1. Dykstra
    let p0 = x0.zeros_like()?;
    let q0 = x0.zeros_like()?;
    let (x1, y1, p1, q1) = opt::step_dykstra(x0, &p0, &q0, normal, offset, 0.0, 2.0)?;
    let diff_x = x1
        .sub(golden_dykstra_x)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_y = y1
        .sub(golden_dykstra_y)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_p = p1
        .sub(golden_dykstra_p)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_q = q1
        .sub(golden_dykstra_q)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_x < 1e-5, "Dykstra x diff: {}", diff_x);
    assert!(diff_y < 1e-5, "Dykstra y diff: {}", diff_y);
    assert!(diff_p < 1e-5, "Dykstra p diff: {}", diff_p);
    assert!(diff_q < 1e-5, "Dykstra q diff: {}", diff_q);

    // 2. BCD
    let p_bcd = opt::step_bcd(param_bcd, b_bcd, a_diag, 0.2)?;
    let diff_bcd = p_bcd
        .sub(golden_param_bcd)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_bcd < 1e-5, "BCD diff: {}", diff_bcd);

    // 3. IRLS
    let p_irls = opt::step_irls(param_irls, target_irls, 0.5, 1e-4)?;
    let diff_irls = p_irls
        .sub(golden_param_irls)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_irls < 1e-5, "IRLS diff: {}", diff_irls);

    // 4. CCCP
    let p_cccp = opt::step_cccp(param_cccp)?;
    let diff_cccp = p_cccp
        .sub(golden_param_cccp)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_cccp < 1e-5, "CCCP diff: {}", diff_cccp);

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
    let module = opt::CoordinateMMUnifiedModule::load_from_safetensors(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["dykstra_err"] < 1e-5);
    assert!(errs["bcd_err"] < 1e-5);
    assert!(errs["irls_err"] < 1e-5);
    assert!(errs["cccp_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
