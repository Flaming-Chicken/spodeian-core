use candle_core::{safetensors::load, Device, Result};

use ai_candle::c04_training_methods_and_optimisers::t01_first_order_and_adaptive_optimisers as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let param = weights.get("param").unwrap();
    let grad = weights.get("grad").unwrap();
    let m_init = weights.get("m_init").unwrap();
    let v_init = weights.get("v_init").unwrap();

    // 1. SGD Momentum
    let (p_sgd, v_sgd) = opt::step_sgd_momentum(param, grad, m_init, 0.05, 0.9, 0.01)?;
    let expected_p_sgd = weights.get("p_sgd").unwrap();
    let expected_v_sgd = weights.get("v_sgd").unwrap();
    let diff_p_sgd = p_sgd
        .sub(expected_p_sgd)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_sgd = v_sgd
        .sub(expected_v_sgd)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_sgd < 1e-5, "p_sgd diff: {}", diff_p_sgd);
    assert!(diff_v_sgd < 1e-5, "v_sgd diff: {}", diff_v_sgd);

    // 2. Adagrad
    let (p_adagrad, g_adagrad) = opt::step_adagrad(param, grad, v_init, 0.05, 1e-8, 0.01)?;
    let expected_p_ada = weights.get("p_adagrad").unwrap();
    let expected_g_ada = weights.get("g_adagrad").unwrap();
    let diff_p_ada = p_adagrad
        .sub(expected_p_ada)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_g_ada = g_adagrad
        .sub(expected_g_ada)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_ada < 1e-5, "p_adagrad diff: {}", diff_p_ada);
    assert!(diff_g_ada < 1e-5, "g_adagrad diff: {}", diff_g_ada);

    // 3. RMSprop
    let (p_rmsprop, v_rmsprop) = opt::step_rmsprop(param, grad, v_init, 0.01, 0.99, 1e-8, 0.01)?;
    let expected_p_rms = weights.get("p_rmsprop").unwrap();
    let expected_v_rms = weights.get("v_rmsprop").unwrap();
    let diff_p_rms = p_rmsprop
        .sub(expected_p_rms)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_rms = v_rmsprop
        .sub(expected_v_rms)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_rms < 1e-5, "p_rmsprop diff: {}", diff_p_rms);
    assert!(diff_v_rms < 1e-5, "v_rmsprop diff: {}", diff_v_rms);

    // 4. Adam
    let (p_adam, m_adam, v_adam) = opt::step_adam(
        param, grad, m_init, v_init, 2, 0.001, 0.9, 0.999, 1e-8, 0.01,
    )?;
    let expected_p_adam = weights.get("p_adam").unwrap();
    let expected_m_adam = weights.get("m_adam").unwrap();
    let expected_v_adam = weights.get("v_adam").unwrap();
    let diff_p_adam = p_adam
        .sub(expected_p_adam)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m_adam = m_adam
        .sub(expected_m_adam)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_adam = v_adam
        .sub(expected_v_adam)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_adam < 1e-5, "p_adam diff: {}", diff_p_adam);
    assert!(diff_m_adam < 1e-5, "m_adam diff: {}", diff_m_adam);
    assert!(diff_v_adam < 1e-5, "v_adam diff: {}", diff_v_adam);

    // 5. AdamW
    let (p_adamw, m_adamw, v_adamw) = opt::step_adamw(
        param, grad, m_init, v_init, 2, 0.001, 0.9, 0.999, 1e-8, 0.01,
    )?;
    let expected_p_adamw = weights.get("p_adamw").unwrap();
    let expected_m_adamw = weights.get("m_adamw").unwrap();
    let expected_v_adamw = weights.get("v_adamw").unwrap();
    let diff_p_adamw = p_adamw
        .sub(expected_p_adamw)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m_adamw = m_adamw
        .sub(expected_m_adamw)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_adamw = v_adamw
        .sub(expected_v_adamw)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_adamw < 1e-5, "p_adamw diff: {}", diff_p_adamw);
    assert!(diff_m_adamw < 1e-5, "m_adamw diff: {}", diff_m_adamw);
    assert!(diff_v_adamw < 1e-5, "v_adamw diff: {}", diff_v_adamw);

    // 6. AdaBelief
    let (p_adab, m_adab, s_adab) = opt::step_adabelief(
        param, grad, m_init, v_init, 2, 0.001, 0.9, 0.999, 1e-16, 0.01,
    )?;
    let expected_p_adab = weights.get("p_adabelief").unwrap();
    let expected_m_adab = weights.get("m_adabelief").unwrap();
    let expected_s_adab = weights.get("s_adabelief").unwrap();
    let diff_p_adab = p_adab
        .sub(expected_p_adab)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m_adab = m_adab
        .sub(expected_m_adab)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_s_adab = s_adab
        .sub(expected_s_adab)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_adab < 1e-5, "p_adab diff: {}", diff_p_adab);
    assert!(diff_m_adab < 1e-5, "m_adab diff: {}", diff_m_adab);
    assert!(diff_s_adab < 1e-5, "s_adab diff: {}", diff_s_adab);

    // 7. Linear Anchor
    let core_w = weights.get("core_weight").unwrap();
    let core_b = weights.get("core_bias").unwrap();
    let anchor = opt::LinearAnchor::new(core_w.clone(), core_b.clone());
    let anchor_x = weights.get("anchor_x").unwrap();
    let anchor_y = anchor.forward(anchor_x)?;
    let expected_anchor_y = weights.get("anchor_y").unwrap();
    let diff_anchor = anchor_y
        .sub(expected_anchor_y)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "anchor_y diff: {}", diff_anchor);

    Ok(())
}
