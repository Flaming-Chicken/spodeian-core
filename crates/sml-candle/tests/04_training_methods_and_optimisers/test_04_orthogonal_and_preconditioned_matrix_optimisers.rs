use candle_core::{Device, Result, safetensors::load};

use sml_candle::c04_training_methods_and_optimisers::t04_orthogonal_and_preconditioned_matrix_optimisers as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let param_2d = weights.get("param_2d").unwrap();
    let grad_2d = weights.get("grad_2d").unwrap();

    // 1. Muon
    let m_init_muon = weights.get("m_init_muon").unwrap();
    let (p_muon, m_muon) =
        opt::step_muon(param_2d, grad_2d, m_init_muon, 0.02, 0.95, true, 0.01, 5)?;
    let expected_p_muon = weights.get("p_muon").unwrap();
    let expected_m_muon = weights.get("m_muon").unwrap();

    let diff_p_muon = p_muon
        .sub(expected_p_muon)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m_muon = m_muon
        .sub(expected_m_muon)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_muon < 1e-5, "p_muon diff: {}", diff_p_muon);
    assert!(diff_m_muon < 1e-5, "m_muon diff: {}", diff_m_muon);

    // 2. Shampoo
    let l_init = weights.get("l_init").unwrap();
    let r_init = weights.get("r_init").unwrap();
    let (p_shampoo, l_shampoo, r_shampoo) =
        opt::step_shampoo(param_2d, grad_2d, l_init, r_init, 0.005, 0.95, 1e-6, 0.01)?;
    let expected_p_shampoo = weights.get("p_shampoo").unwrap();
    let expected_l_shampoo = weights.get("l_shampoo").unwrap();
    let expected_r_shampoo = weights.get("r_shampoo").unwrap();

    let diff_p_shampoo = p_shampoo
        .sub(expected_p_shampoo)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_l_shampoo = l_shampoo
        .sub(expected_l_shampoo)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_r_shampoo = r_shampoo
        .sub(expected_r_shampoo)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_shampoo < 1e-5, "p_shampoo diff: {}", diff_p_shampoo);
    assert!(diff_l_shampoo < 1e-5, "l_shampoo diff: {}", diff_l_shampoo);
    assert!(diff_r_shampoo < 1e-5, "r_shampoo diff: {}", diff_r_shampoo);

    // 3. GaLore
    let m_init_galore = weights.get("m_init_galore").unwrap();
    let v_init_galore = weights.get("v_init_galore").unwrap();
    let proj_matrix = weights.get("proj_matrix").unwrap();

    let (p_galore, m_galore, v_galore) = opt::step_galore(
        param_2d,
        grad_2d,
        m_init_galore,
        v_init_galore,
        proj_matrix,
        2,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
    )?;
    let expected_p_galore = weights.get("p_galore").unwrap();
    let expected_m_galore = weights.get("m_galore").unwrap();
    let expected_v_galore = weights.get("v_galore").unwrap();

    let diff_p_galore = p_galore
        .sub(expected_p_galore)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_m_galore = m_galore
        .sub(expected_m_galore)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_v_galore = v_galore
        .sub(expected_v_galore)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_p_galore < 1e-5, "p_galore diff: {}", diff_p_galore);
    assert!(diff_m_galore < 1e-5, "m_galore diff: {}", diff_m_galore);
    assert!(diff_v_galore < 1e-5, "v_galore diff: {}", diff_v_galore);

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
