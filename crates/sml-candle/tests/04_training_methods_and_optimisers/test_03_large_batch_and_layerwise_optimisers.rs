use candle_core::{Device, Result, safetensors::load};

use sml_candle::c04_training_methods_and_optimisers::t03_large_batch_and_layerwise_optimisers as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.safetensors",
    );
    let weights = load(weights_path, &device)?;

    // 1. LARS
    let param = weights.get("param").unwrap();
    let grad = weights.get("grad").unwrap();
    let v_init_lars = weights.get("v_init_lars").unwrap();

    let (p_lars, v_lars, ratio_lars) =
        opt::step_lars(param, grad, v_init_lars, 0.1, 0.9, 0.0001, 0.001, 1e-8)?;
    let expected_p_lars = weights.get("p_lars").unwrap();
    let expected_v_lars = weights.get("v_lars").unwrap();
    let expected_ratio_lars = weights.get("ratio_lars").unwrap();

    let diff_p_lars = p_lars
        .sub(expected_p_lars)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_lars = v_lars
        .sub(expected_v_lars)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_ratio_lars = ratio_lars
        .sub(expected_ratio_lars)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_lars < 1e-5, "p_lars diff: {}", diff_p_lars);
    assert!(diff_v_lars < 1e-5, "v_lars diff: {}", diff_v_lars);
    assert!(
        diff_ratio_lars < 1e-5,
        "ratio_lars diff: {}",
        diff_ratio_lars
    );

    // 2. LAMB
    let m_init_lamb = weights.get("m_init_lamb").unwrap();
    let v_init_lamb = weights.get("v_init_lamb").unwrap();

    let (p_lamb, m_lamb, v_lamb, ratio_lamb) = opt::step_lamb(
        param,
        grad,
        m_init_lamb,
        v_init_lamb,
        2,
        0.001,
        0.9,
        0.999,
        1e-6,
        0.01,
    )?;
    let expected_p_lamb = weights.get("p_lamb").unwrap();
    let expected_m_lamb = weights.get("m_lamb").unwrap();
    let expected_v_lamb = weights.get("v_lamb").unwrap();
    let expected_ratio_lamb = weights.get("ratio_lamb").unwrap();

    let diff_p_lamb = p_lamb
        .sub(expected_p_lamb)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m_lamb = m_lamb
        .sub(expected_m_lamb)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_lamb = v_lamb
        .sub(expected_v_lamb)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_ratio_lamb = ratio_lamb
        .sub(expected_ratio_lamb)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_lamb < 1e-5, "p_lamb diff: {}", diff_p_lamb);
    assert!(diff_m_lamb < 1e-5, "m_lamb diff: {}", diff_m_lamb);
    assert!(diff_v_lamb < 1e-5, "v_lamb diff: {}", diff_v_lamb);
    assert!(
        diff_ratio_lamb < 1e-5,
        "ratio_lamb diff: {}",
        diff_ratio_lamb
    );

    // 3. Gradient Noise Scale
    let per_sample_grads = weights.get("per_sample_grads").unwrap();
    let (g_norm_sq, trace_sigma, b_simple) = opt::compute_gradient_noise_scale(per_sample_grads)?;

    let expected_g_norm_sq = weights.get("g_norm_sq").unwrap();
    let expected_trace_sigma = weights.get("trace_sigma").unwrap();
    let expected_b_simple = weights.get("b_simple").unwrap();

    let diff_g = g_norm_sq
        .sub(expected_g_norm_sq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_trace = trace_sigma
        .sub(expected_trace_sigma)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_b = b_simple
        .sub(expected_b_simple)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_g < 1e-5, "g_norm_sq diff: {}", diff_g);
    assert!(diff_trace < 1e-5, "trace_sigma diff: {}", diff_trace);
    assert!(diff_b < 1e-5, "b_simple diff: {}", diff_b);

    // 4. Linear Anchor
    let core_w = weights.get("weight").unwrap();
    let core_b = weights.get("bias").unwrap();
    let anchor = opt::LinearAnchor::new(core_w.clone(), core_b.clone());
    let test_in = weights.get("test_input").unwrap();
    let pred_y = anchor.forward(test_in)?;
    let expected_y = weights.get("test_output").unwrap();

    let diff_anchor = pred_y
        .sub(expected_y)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "anchor diff: {}", diff_anchor);

    Ok(())
}
