use candle_core::{Device, Result, Tensor, safetensors::load};

use ai_candle::c04_training_methods_and_optimisers::t07_learning_rate_schedulers_and_lr_finder as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let param_cosine = weights.get("param_cosine").unwrap();
    let grad_cosine = weights.get("grad_cosine").unwrap();
    let golden_lr_cosine = weights.get("golden_lr_cosine").unwrap().to_vec1::<f32>()?[0] as f64;
    let golden_param_cosine = weights.get("golden_param_cosine").unwrap();

    let param_wsd = weights.get("param_wsd").unwrap();
    let grad_wsd = weights.get("grad_wsd").unwrap();
    let golden_lr_wsd = weights.get("golden_lr_wsd").unwrap().to_vec1::<f32>()?[0] as f64;
    let golden_param_wsd = weights.get("golden_param_wsd").unwrap();

    let param_onecycle = weights.get("param_onecycle").unwrap();
    let grad_onecycle = weights.get("grad_onecycle").unwrap();
    let golden_lr_onecycle = weights
        .get("golden_lr_onecycle")
        .unwrap()
        .to_vec1::<f32>()?[0] as f64;
    let golden_param_onecycle = weights.get("golden_param_onecycle").unwrap();

    let param_autoclip = weights.get("param_autoclip").unwrap();
    let grad_autoclip = weights.get("grad_autoclip").unwrap();
    let norm_history = weights.get("norm_history").unwrap();
    let golden_clipped_grad = weights.get("golden_clipped_grad").unwrap();
    let golden_param_autoclip = weights.get("golden_param_autoclip").unwrap();

    // 1. Cosine Annealing with Warmup
    let lr_cos = opt::schedule_cosine_warmup(30, 100, 10, 1e-5, 0.01);
    assert!((lr_cos - golden_lr_cosine).abs() < 1e-5);
    let p_cos_vec: Vec<f32> = param_cosine
        .to_vec1::<f32>()?
        .iter()
        .zip(grad_cosine.to_vec1::<f32>()?.iter())
        .map(|(&p, &g)| p - (lr_cos as f32) * g)
        .collect();
    let p_cos = Tensor::from_vec(p_cos_vec, param_cosine.dim(0)?, param_cosine.device())?;
    let diff_cos = p_cos
        .sub(golden_param_cosine)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_cos < 1e-5, "cosine diff: {}", diff_cos);

    // 2. WSD
    let lr_wsd = opt::schedule_wsd(85, 100, 10, 20, 1e-4, 0.01);
    assert!((lr_wsd - golden_lr_wsd).abs() < 1e-5);
    let p_wsd_vec: Vec<f32> = param_wsd
        .to_vec1::<f32>()?
        .iter()
        .zip(grad_wsd.to_vec1::<f32>()?.iter())
        .map(|(&p, &g)| p - (lr_wsd as f32) * g)
        .collect();
    let p_wsd = Tensor::from_vec(p_wsd_vec, param_wsd.dim(0)?, param_wsd.device())?;
    let diff_wsd = p_wsd
        .sub(golden_param_wsd)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_wsd < 1e-5, "wsd diff: {}", diff_wsd);

    // 3. OneCycleLR
    let lr_oc = opt::schedule_onecycle(40, 100, 30, 0.01, 25.0, 1000.0);
    assert!((lr_oc - golden_lr_onecycle).abs() < 1e-5);
    let p_oc_vec: Vec<f32> = param_onecycle
        .to_vec1::<f32>()?
        .iter()
        .zip(grad_onecycle.to_vec1::<f32>()?.iter())
        .map(|(&p, &g)| p - (lr_oc as f32) * g)
        .collect();
    let p_oc = Tensor::from_vec(p_oc_vec, param_onecycle.dim(0)?, param_onecycle.device())?;
    let diff_oc = p_oc
        .sub(golden_param_onecycle)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_oc < 1e-5, "onecycle diff: {}", diff_oc);

    // 4. AutoClip
    let (p_ac, clipped_g, _) = opt::step_autoclip(
        param_autoclip,
        grad_autoclip,
        norm_history,
        90.0,
        0.01,
        1e-8,
    )?;
    let diff_cg = clipped_g
        .sub(golden_clipped_grad)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_p_ac = p_ac
        .sub(golden_param_autoclip)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_cg < 1e-5, "clipped_grad diff: {}", diff_cg);
    assert!(diff_p_ac < 1e-5, "autoclip param diff: {}", diff_p_ac);

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
    let module = opt::LRSchedulersUnifiedModule::load_from_safetensors(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["cosine_err"] < 1e-5);
    assert!(errs["wsd_err"] < 1e-5);
    assert!(errs["onecycle_err"] < 1e-5);
    assert!(errs["autoclip_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
