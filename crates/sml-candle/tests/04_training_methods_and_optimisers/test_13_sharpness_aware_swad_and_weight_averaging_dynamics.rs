use candle_core::{Device, Result, safetensors::load};

use sml_candle::c04_training_methods_and_optimisers::t13_sharpness_aware_swad_and_weight_averaging_dynamics as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let param_sam = weights.get("param_sam").unwrap();
    let grad_at_w = weights.get("grad_at_w").unwrap();
    let grad_at_pert = weights.get("grad_at_pert").unwrap();
    let golden_param_sam = weights.get("golden_param_sam").unwrap();
    let golden_epsilon_sam = weights.get("golden_epsilon_sam").unwrap();

    let slow_p = weights.get("slow_p").unwrap();
    let fast_p = weights.get("fast_p").unwrap();
    let golden_slow_p = weights.get("golden_slow_p").unwrap();
    let golden_fast_p = weights.get("golden_fast_p").unwrap();

    let swa_avg = weights.get("swa_avg").unwrap();
    let ema_avg = weights.get("ema_avg").unwrap();
    let curr_param = weights.get("curr_param").unwrap();
    let golden_swa = weights.get("golden_swa").unwrap();
    let golden_ema = weights.get("golden_ema").unwrap();

    let soup_models = weights.get("soup_models").unwrap();
    let golden_soup = weights.get("golden_soup").unwrap();

    let anchor_w = weights.get("anchor_w").unwrap();
    let anchor_b = weights.get("anchor_b").unwrap();
    let anchor_x = weights.get("anchor_x").unwrap();
    let golden_anchor_out = weights.get("golden_anchor_out").unwrap();

    // 1. SAM Invariant
    let (p_sam, eps_sam) =
        opt::step_sam(param_sam, grad_at_w, grad_at_pert, 0.05, 0.01, 0.001, 1e-12)?;
    let diff_p = p_sam
        .sub(golden_param_sam)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_eps = eps_sam
        .sub(golden_epsilon_sam)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p < 1e-5, "SAM param diff: {}", diff_p);
    assert!(diff_eps < 1e-5, "SAM epsilon diff: {}", diff_eps);

    // 2. Lookahead Invariant
    let (slow_out, fast_out) = opt::step_lookahead(slow_p, fast_p, 0.5)?;
    let diff_slow = slow_out
        .sub(golden_slow_p)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_fast = fast_out
        .sub(golden_fast_p)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_slow < 1e-5, "Lookahead slow diff: {}", diff_slow);
    assert!(diff_fast < 1e-5, "Lookahead fast diff: {}", diff_fast);

    // 3. SWA & EMA Invariant
    let (new_swa, new_ema) = opt::step_swa_ema(swa_avg, ema_avg, curr_param, 4, 0.99)?;
    let diff_swa = new_swa
        .sub(golden_swa)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_ema = new_ema
        .sub(golden_ema)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_swa < 1e-5, "SWA diff: {}", diff_swa);
    assert!(diff_ema < 1e-5, "EMA diff: {}", diff_ema);

    // 4. Model Soup Invariant
    let soup_out = opt::uniform_model_soup(soup_models)?;
    let diff_soup = soup_out
        .sub(golden_soup)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_soup < 1e-5, "Model soup diff: {}", diff_soup);

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
    let module = opt::SharpnessAwareUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["sam_err"] < 1e-5);
    assert!(errs["lookahead_err"] < 1e-5);
    assert!(errs["swa_ema_err"] < 1e-5);
    assert!(errs["soup_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
