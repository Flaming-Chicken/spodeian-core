use candle_core::{Device, Result, safetensors::load};

use ai_candle::c04_training_methods_and_optimisers::t14_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_14_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let matrix_a = weights.get("matrix_A").unwrap();
    let state_x = weights.get("state_x").unwrap();
    let state_y = weights.get("state_y").unwrap();
    let state_x_prev = weights.get("state_x_prev").unwrap();
    let state_y_prev = weights.get("state_y_prev").unwrap();

    let golden_x_eg = weights.get("golden_x_eg").unwrap();
    let golden_y_eg = weights.get("golden_y_eg").unwrap();
    let golden_x_ogda = weights.get("golden_x_ogda").unwrap();
    let golden_y_ogda = weights.get("golden_y_ogda").unwrap();
    let golden_x_sga = weights.get("golden_x_sga").unwrap();
    let golden_y_sga = weights.get("golden_y_sga").unwrap();

    let x_best = weights.get("x_best").unwrap();
    let x_second = weights.get("x_second").unwrap();
    let x_worst = weights.get("x_worst").unwrap();
    let golden_x_reflect = weights.get("golden_x_reflect").unwrap();

    let anchor_w = weights.get("anchor_w").unwrap();
    let anchor_b = weights.get("anchor_b").unwrap();
    let anchor_x = weights.get("anchor_x").unwrap();
    let golden_anchor_out = weights.get("golden_anchor_out").unwrap();

    // 1. Extragradient Invariant
    let (x_eg, y_eg) = opt::step_extragradient(state_x, state_y, matrix_a, 0.1, 0.1)?;
    let diff_x_eg = x_eg
        .sub(golden_x_eg)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_y_eg = y_eg
        .sub(golden_y_eg)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_x_eg < 1e-5, "Extragradient x diff: {}", diff_x_eg);
    assert!(diff_y_eg < 1e-5, "Extragradient y diff: {}", diff_y_eg);

    // 2. OGDA Invariant
    let (x_ogda, y_ogda) = opt::step_ogda(
        state_x,
        state_y,
        state_x_prev,
        state_y_prev,
        matrix_a,
        0.1,
        0.1,
    )?;
    let diff_x_ogda = x_ogda
        .sub(golden_x_ogda)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_y_ogda = y_ogda
        .sub(golden_y_ogda)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_x_ogda < 1e-5, "OGDA x diff: {}", diff_x_ogda);
    assert!(diff_y_ogda < 1e-5, "OGDA y diff: {}", diff_y_ogda);

    // 3. SGA Invariant
    let (x_sga, y_sga) = opt::step_sga(state_x, state_y, matrix_a, 0.5, 0.1, 0.1)?;
    let diff_x_sga = x_sga
        .sub(golden_x_sga)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_y_sga = y_sga
        .sub(golden_y_sga)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_x_sga < 1e-5, "SGA x diff: {}", diff_x_sga);
    assert!(diff_y_sga < 1e-5, "SGA y diff: {}", diff_y_sga);

    // 4. Nelder-Mead Reflection Invariant
    let x_refl = opt::nelder_mead_reflect(x_best, x_second, x_worst, 1.0)?;
    let diff_refl = x_refl
        .sub(golden_x_reflect)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_refl < 1e-5, "Nelder-Mead diff: {}", diff_refl);

    // 5. Linear Anchor Invariant
    let anchor = opt::LinearAnchor::new(anchor_w.clone(), anchor_b.clone());
    let anchor_out = anchor.forward(anchor_x)?;
    let diff_anchor = anchor_out
        .sub(golden_anchor_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "Linear anchor diff: {}", diff_anchor);

    // Module verify_invariants
    let module = opt::MinimaxUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["extragradient_err"] < 1e-5);
    assert!(errs["ogda_err"] < 1e-5);
    assert!(errs["sga_err"] < 1e-5);
    assert!(errs["nelder_mead_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
