use candle_core::{Device, Result, safetensors::load};

use sml_candle::c04_training_methods_and_optimisers::t02_modern_sign_and_factored_optimisers as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.safetensors",
    );
    let weights = load(weights_path, &device)?;

    // 1. Lion
    let param_1d = weights.get("param_1d").unwrap();
    let grad_1d = weights.get("grad_1d").unwrap();
    let m_1d = weights.get("m_1d").unwrap();

    let (p_lion, m_lion) = opt::step_lion(param_1d, grad_1d, m_1d, 0.0001, 0.9, 0.99, 0.1)?;
    let expected_p_lion = weights.get("p_lion").unwrap();
    let expected_m_lion = weights.get("m_lion").unwrap();

    let diff_p_lion = p_lion
        .sub(expected_p_lion)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m_lion = m_lion
        .sub(expected_m_lion)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_lion < 1e-5, "p_lion diff: {}", diff_p_lion);
    assert!(diff_m_lion < 1e-5, "m_lion diff: {}", diff_m_lion);

    // 2. Adafactor (2D)
    let param_2d = weights.get("param_2d").unwrap();
    let grad_2d = weights.get("grad_2d").unwrap();
    let r_buf = weights.get("r_buf").unwrap();
    let c_buf = weights.get("c_buf").unwrap();

    let (p_adafactor, r_adafactor, c_adafactor) = opt::step_adafactor_2d(
        param_2d, grad_2d, r_buf, c_buf, 2, 0.001, 0.8, 1e-30, 1e-3, 1.0, 0.01,
    )?;
    let expected_p_ada = weights.get("p_adafactor").unwrap();
    let expected_r_ada = weights.get("r_adafactor").unwrap();
    let expected_c_ada = weights.get("c_adafactor").unwrap();

    let diff_p_ada = p_adafactor
        .sub(expected_p_ada)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_r_ada = r_adafactor
        .sub(expected_r_ada)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_c_ada = c_adafactor
        .sub(expected_c_ada)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_ada < 1e-5, "p_adafactor diff: {}", diff_p_ada);
    assert!(diff_r_ada < 1e-5, "r_adafactor diff: {}", diff_r_ada);
    assert!(diff_c_ada < 1e-5, "c_adafactor diff: {}", diff_c_ada);

    // 3. AdEMAMix
    let m2_1d = weights.get("m2_1d").unwrap();
    let v_1d = weights.get("v_1d").unwrap();

    let (p_adem, m1_adem, m2_adem, v_adem) = opt::step_ademamix(
        param_1d, grad_1d, m_1d, m2_1d, v_1d, 2, 0.001, 0.9, 0.999, 0.9999, 5.0, 1e-8, 0.01,
    )?;
    let expected_p_adem = weights.get("p_ademamix").unwrap();
    let expected_m1_adem = weights.get("m1_ademamix").unwrap();
    let expected_m2_adem = weights.get("m2_ademamix").unwrap();
    let expected_v_adem = weights.get("v_ademamix").unwrap();

    let diff_p_adem = p_adem
        .sub(expected_p_adem)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m1_adem = m1_adem
        .sub(expected_m1_adem)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m2_adem = m2_adem
        .sub(expected_m2_adem)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_adem = v_adem
        .sub(expected_v_adem)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_adem < 1e-5, "p_adem diff: {}", diff_p_adem);
    assert!(diff_m1_adem < 1e-5, "m1_adem diff: {}", diff_m1_adem);
    assert!(diff_m2_adem < 1e-5, "m2_adem diff: {}", diff_m2_adem);
    assert!(diff_v_adem < 1e-5, "v_adem diff: {}", diff_v_adem);

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
