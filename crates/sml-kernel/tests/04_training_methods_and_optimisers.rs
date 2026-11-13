//! Category 04 Training Methods & Optimisers Integration Test Suite for ai-kernel.

use ai_kernel::KernelDevice;
use ai_kernel::c04_training_methods_and_optimisers::t01_first_order_and_adaptive_optimisers as t01;
use candle_core::{Device, Result, safetensors::load};

#[path = "common/mod.rs"]
mod common;

#[test]
fn test_01_first_order_and_adaptive_optimisers_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param = tensors.get("param").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad").unwrap().to_vec1::<f32>()?;
    let m_init = tensors.get("m_init").unwrap().to_vec1::<f32>()?;
    let v_init = tensors.get("v_init").unwrap().to_vec1::<f32>()?;

    let gold_p_adamw = tensors.get("p_adamw").unwrap().to_vec1::<f32>()?;
    let gold_m_adamw = tensors.get("m_adamw").unwrap().to_vec1::<f32>()?;
    let gold_v_adamw = tensors.get("v_adamw").unwrap().to_vec1::<f32>()?;

    // 1. CPU Fused AdamW Step
    let n = param.len();
    let mut cpu_p = vec![0.0f32; n];
    let mut cpu_m = vec![0.0f32; n];
    let mut cpu_v = vec![0.0f32; n];

    t01::adamw_cpu(
        &param, &grad, &m_init, &v_init, &mut cpu_p, &mut cpu_m, &mut cpu_v, 2, 0.001, 0.01, 0.9,
        0.999, 1e-8,
    );

    for i in 0..n {
        assert!(
            (cpu_p[i] - gold_p_adamw[i]).abs() < 1e-5,
            "CPU p_adamw error at {i}: {} vs {}",
            cpu_p[i],
            gold_p_adamw[i]
        );
        assert!(
            (cpu_m[i] - gold_m_adamw[i]).abs() < 1e-5,
            "CPU m_adamw error at {i}: {} vs {}",
            cpu_m[i],
            gold_m_adamw[i]
        );
        assert!(
            (cpu_v[i] - gold_v_adamw[i]).abs() < 1e-5,
            "CPU v_adamw error at {i}: {} vs {}",
            cpu_v[i],
            gold_v_adamw[i]
        );
    }

    // 2. Auto-dispatcher (WGPU with CPU fallback)
    let kdev = KernelDevice::Cpu;
    let (auto_p, auto_m, auto_v) = t01::adamw_auto(
        &kdev, &param, &grad, &m_init, &v_init, 2, 0.001, 0.01, 0.9, 0.999, 1e-8,
    );

    for i in 0..n {
        assert!(
            (auto_p[i] - gold_p_adamw[i]).abs() < 1e-5,
            "Auto p_adamw error at {i}: {} vs {}",
            auto_p[i],
            gold_p_adamw[i]
        );
        assert!(
            (auto_m[i] - gold_m_adamw[i]).abs() < 1e-5,
            "Auto m_adamw error at {i}: {} vs {}",
            auto_m[i],
            gold_m_adamw[i]
        );
        assert!(
            (auto_v[i] - gold_v_adamw[i]).abs() < 1e-5,
            "Auto v_adamw error at {i}: {} vs {}",
            auto_v[i],
            gold_v_adamw[i]
        );
    }

    Ok(())
}

#[test]
fn test_02_modern_sign_and_factored_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t02_modern_sign_and_factored_optimisers as t02;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param = tensors.get("param_1d").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad_1d").unwrap().to_vec1::<f32>()?;
    let m_init = tensors.get("m_1d").unwrap().to_vec1::<f32>()?;

    let gold_p_lion = tensors.get("p_lion").unwrap().to_vec1::<f32>()?;
    let gold_m_lion = tensors.get("m_lion").unwrap().to_vec1::<f32>()?;

    // 1. CPU Fused Lion Step
    let n = param.len();
    let mut cpu_p = vec![0.0f32; n];
    let mut cpu_m = vec![0.0f32; n];

    t02::lion_cpu(
        &param, &grad, &m_init, &mut cpu_p, &mut cpu_m, 0.0001, 0.1, 0.9, 0.99,
    );

    for i in 0..n {
        assert!(
            (cpu_p[i] - gold_p_lion[i]).abs() < 1e-5,
            "CPU p_lion error at {i}: {} vs {}",
            cpu_p[i],
            gold_p_lion[i]
        );
        assert!(
            (cpu_m[i] - gold_m_lion[i]).abs() < 1e-5,
            "CPU m_lion error at {i}: {} vs {}",
            cpu_m[i],
            gold_m_lion[i]
        );
    }

    // 2. Auto-dispatcher (WGPU with CPU fallback)
    let kdev = KernelDevice::Cpu;
    let (auto_p, auto_m) = t02::lion_auto(&kdev, &param, &grad, &m_init, 0.0001, 0.1, 0.9, 0.99);

    for i in 0..n {
        assert!(
            (auto_p[i] - gold_p_lion[i]).abs() < 1e-5,
            "Auto p_lion error at {i}: {} vs {}",
            auto_p[i],
            gold_p_lion[i]
        );
        assert!(
            (auto_m[i] - gold_m_lion[i]).abs() < 1e-5,
            "Auto m_lion error at {i}: {} vs {}",
            auto_m[i],
            gold_m_lion[i]
        );
    }

    Ok(())
}

#[test]
fn test_03_large_batch_and_layerwise_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t03_large_batch_and_layerwise_optimisers as t03;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param = tensors.get("param").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad").unwrap().to_vec1::<f32>()?;
    let v_init_lars = tensors.get("v_init_lars").unwrap().to_vec1::<f32>()?;
    let m_init_lamb = tensors.get("m_init_lamb").unwrap().to_vec1::<f32>()?;
    let v_init_lamb = tensors.get("v_init_lamb").unwrap().to_vec1::<f32>()?;

    let gold_p_lars = tensors.get("p_lars").unwrap().to_vec1::<f32>()?;
    let gold_v_lars = tensors.get("v_lars").unwrap().to_vec1::<f32>()?;
    let gold_ratio_lars = tensors.get("ratio_lars").unwrap().to_vec1::<f32>()?[0];

    let gold_p_lamb = tensors.get("p_lamb").unwrap().to_vec1::<f32>()?;
    let gold_m_lamb = tensors.get("m_lamb").unwrap().to_vec1::<f32>()?;
    let gold_v_lamb = tensors.get("v_lamb").unwrap().to_vec1::<f32>()?;
    let gold_ratio_lamb = tensors.get("ratio_lamb").unwrap().to_vec1::<f32>()?[0];

    let n = param.len();

    // 1. LARS CPU & Auto
    let mut cpu_p_lars = vec![0.0f32; n];
    let mut cpu_v_lars = vec![0.0f32; n];
    let cpu_ratio_lars = t03::lars_cpu(
        &param,
        &grad,
        &v_init_lars,
        &mut cpu_p_lars,
        &mut cpu_v_lars,
        0.1,
        0.9,
        0.0001,
        0.001,
        1e-8,
    );
    assert!((cpu_ratio_lars - gold_ratio_lars).abs() < 1e-5);
    for i in 0..n {
        assert!((cpu_p_lars[i] - gold_p_lars[i]).abs() < 1e-5);
        assert!((cpu_v_lars[i] - gold_v_lars[i]).abs() < 1e-5);
    }

    let kdev = KernelDevice::Cpu;
    let (auto_p_lars, auto_v_lars, auto_ratio_lars) = t03::lars_auto(
        &kdev,
        &param,
        &grad,
        &v_init_lars,
        0.1,
        0.9,
        0.0001,
        0.001,
        1e-8,
    );
    assert!((auto_ratio_lars - gold_ratio_lars).abs() < 1e-5);
    for i in 0..n {
        assert!((auto_p_lars[i] - gold_p_lars[i]).abs() < 1e-5);
        assert!((auto_v_lars[i] - gold_v_lars[i]).abs() < 1e-5);
    }

    // 2. LAMB CPU & Auto
    let mut cpu_p_lamb = vec![0.0f32; n];
    let mut cpu_m_lamb = vec![0.0f32; n];
    let mut cpu_v_lamb = vec![0.0f32; n];
    let cpu_ratio_lamb = t03::lamb_cpu(
        &param,
        &grad,
        &m_init_lamb,
        &v_init_lamb,
        &mut cpu_p_lamb,
        &mut cpu_m_lamb,
        &mut cpu_v_lamb,
        2,
        0.001,
        0.9,
        0.999,
        1e-6,
        0.01,
    );
    assert!((cpu_ratio_lamb - gold_ratio_lamb).abs() < 2e-4);
    for i in 0..n {
        assert!((cpu_p_lamb[i] - gold_p_lamb[i]).abs() < 1e-5);
        assert!((cpu_m_lamb[i] - gold_m_lamb[i]).abs() < 1e-5);
        assert!((cpu_v_lamb[i] - gold_v_lamb[i]).abs() < 1e-5);
    }

    let (auto_p_lamb, auto_m_lamb, auto_v_lamb, auto_ratio_lamb) = t03::lamb_auto(
        &kdev,
        &param,
        &grad,
        &m_init_lamb,
        &v_init_lamb,
        2,
        0.001,
        0.9,
        0.999,
        1e-6,
        0.01,
    );
    assert!((auto_ratio_lamb - gold_ratio_lamb).abs() < 2e-4);
    for i in 0..n {
        assert!((auto_p_lamb[i] - gold_p_lamb[i]).abs() < 1e-5);
        assert!((auto_m_lamb[i] - gold_m_lamb[i]).abs() < 1e-5);
        assert!((auto_v_lamb[i] - gold_v_lamb[i]).abs() < 1e-5);
    }

    // 3. Gradient Noise Scale CPU
    let per_sample_grads = tensors.get("per_sample_grads").unwrap().to_vec2::<f32>()?;
    let flat_grads: Vec<f32> = per_sample_grads.iter().flatten().copied().collect();
    let (g_sq, tr_sig, b_sim) = t03::gradient_noise_scale_cpu(&flat_grads, 8, 4);

    let gold_g_sq = tensors.get("g_norm_sq").unwrap().to_vec1::<f32>()?[0];
    let gold_tr_sig = tensors.get("trace_sigma").unwrap().to_vec1::<f32>()?[0];
    let gold_b_sim = tensors.get("b_simple").unwrap().to_vec1::<f32>()?[0];

    assert!((g_sq - gold_g_sq).abs() < 1e-5);
    assert!((tr_sig - gold_tr_sig).abs() < 1e-5);
    assert!((b_sim - gold_b_sim).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_04_orthogonal_and_preconditioned_matrix_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t04_orthogonal_and_preconditioned_matrix_optimisers as t04;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param_2d = tensors.get("param_2d").unwrap().to_vec2::<f32>()?;
    let grad_2d = tensors.get("grad_2d").unwrap().to_vec2::<f32>()?;
    let flat_p: Vec<f32> = param_2d.iter().flatten().copied().collect();
    let flat_g: Vec<f32> = grad_2d.iter().flatten().copied().collect();

    let m = 4;
    let n = 4;
    let len = m * n;

    // 1. Muon CPU & Auto
    let m_init_muon = tensors.get("m_init_muon").unwrap().to_vec2::<f32>()?;
    let flat_m_muon: Vec<f32> = m_init_muon.iter().flatten().copied().collect();

    let gold_p_muon = tensors.get("p_muon").unwrap().to_vec2::<f32>()?;
    let gold_m_muon = tensors.get("m_muon").unwrap().to_vec2::<f32>()?;
    let flat_gold_p_muon: Vec<f32> = gold_p_muon.iter().flatten().copied().collect();
    let flat_gold_m_muon: Vec<f32> = gold_m_muon.iter().flatten().copied().collect();

    let mut cpu_p_muon = vec![0.0f32; len];
    let mut cpu_m_muon = vec![0.0f32; len];
    t04::muon_cpu(
        &flat_p,
        &flat_g,
        &flat_m_muon,
        &mut cpu_p_muon,
        &mut cpu_m_muon,
        m,
        n,
        0.02,
        0.95,
        true,
        0.01,
        5,
    );
    for i in 0..len {
        assert!((cpu_p_muon[i] - flat_gold_p_muon[i]).abs() < 1e-5);
        assert!((cpu_m_muon[i] - flat_gold_m_muon[i]).abs() < 1e-5);
    }

    let kdev = KernelDevice::Cpu;
    let (auto_p_muon, auto_m_muon) = t04::muon_auto(
        &kdev,
        &flat_p,
        &flat_g,
        &flat_m_muon,
        m,
        n,
        0.02,
        0.95,
        true,
        0.01,
        5,
    );
    for i in 0..len {
        assert!((auto_p_muon[i] - flat_gold_p_muon[i]).abs() < 1e-5);
        assert!((auto_m_muon[i] - flat_gold_m_muon[i]).abs() < 1e-5);
    }

    // 2. Shampoo CPU
    let l_init = tensors.get("l_init").unwrap().to_vec2::<f32>()?;
    let r_init = tensors.get("r_init").unwrap().to_vec2::<f32>()?;
    let flat_l_init: Vec<f32> = l_init.iter().flatten().copied().collect();
    let flat_r_init: Vec<f32> = r_init.iter().flatten().copied().collect();

    let gold_p_shampoo = tensors.get("p_shampoo").unwrap().to_vec2::<f32>()?;
    let gold_l_shampoo = tensors.get("l_shampoo").unwrap().to_vec2::<f32>()?;
    let gold_r_shampoo = tensors.get("r_shampoo").unwrap().to_vec2::<f32>()?;
    let flat_gold_p_shampoo: Vec<f32> = gold_p_shampoo.iter().flatten().copied().collect();
    let flat_gold_l_shampoo: Vec<f32> = gold_l_shampoo.iter().flatten().copied().collect();
    let flat_gold_r_shampoo: Vec<f32> = gold_r_shampoo.iter().flatten().copied().collect();

    let mut cpu_p_shampoo = vec![0.0f32; len];
    let mut cpu_l_shampoo = vec![0.0f32; m * m];
    let mut cpu_r_shampoo = vec![0.0f32; n * n];
    t04::shampoo_cpu(
        &flat_p,
        &flat_g,
        &flat_l_init,
        &flat_r_init,
        &mut cpu_p_shampoo,
        &mut cpu_l_shampoo,
        &mut cpu_r_shampoo,
        m,
        n,
        0.005,
        0.95,
        1e-6,
        0.01,
    );
    for i in 0..len {
        assert!((cpu_p_shampoo[i] - flat_gold_p_shampoo[i]).abs() < 1e-5);
    }
    for i in 0..(m * m) {
        assert!((cpu_l_shampoo[i] - flat_gold_l_shampoo[i]).abs() < 1e-5);
    }
    for i in 0..(n * n) {
        assert!((cpu_r_shampoo[i] - flat_gold_r_shampoo[i]).abs() < 1e-5);
    }

    // 3. GaLore CPU
    let r = 2;
    let m_init_galore = tensors.get("m_init_galore").unwrap().to_vec2::<f32>()?;
    let v_init_galore = tensors.get("v_init_galore").unwrap().to_vec2::<f32>()?;
    let proj = tensors.get("proj_matrix").unwrap().to_vec2::<f32>()?;
    let flat_m_galore: Vec<f32> = m_init_galore.iter().flatten().copied().collect();
    let flat_v_galore: Vec<f32> = v_init_galore.iter().flatten().copied().collect();
    let flat_proj: Vec<f32> = proj.iter().flatten().copied().collect();

    let gold_p_galore = tensors.get("p_galore").unwrap().to_vec2::<f32>()?;
    let gold_m_galore = tensors.get("m_galore").unwrap().to_vec2::<f32>()?;
    let gold_v_galore = tensors.get("v_galore").unwrap().to_vec2::<f32>()?;
    let flat_gold_p_galore: Vec<f32> = gold_p_galore.iter().flatten().copied().collect();
    let flat_gold_m_galore: Vec<f32> = gold_m_galore.iter().flatten().copied().collect();
    let flat_gold_v_galore: Vec<f32> = gold_v_galore.iter().flatten().copied().collect();

    let mut cpu_p_galore = vec![0.0f32; len];
    let mut cpu_m_galore = vec![0.0f32; r * n];
    let mut cpu_v_galore = vec![0.0f32; r * n];
    t04::galore_cpu(
        &flat_p,
        &flat_g,
        &flat_m_galore,
        &flat_v_galore,
        &flat_proj,
        &mut cpu_p_galore,
        &mut cpu_m_galore,
        &mut cpu_v_galore,
        m,
        n,
        r,
        2,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
    );
    for i in 0..len {
        assert!((cpu_p_galore[i] - flat_gold_p_galore[i]).abs() < 1e-5);
    }
    for i in 0..(r * n) {
        assert!((cpu_m_galore[i] - flat_gold_m_galore[i]).abs() < 1e-5);
        assert!((cpu_v_galore[i] - flat_gold_v_galore[i]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_05_second_order_and_curvature_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t05_second_order_and_curvature_optimisers as t05;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let theta = tensors.get("theta").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad").unwrap().to_vec1::<f32>()?;
    let momentum = tensors.get("momentum").unwrap().to_vec1::<f32>()?;
    let hess_diag = tensors.get("hess_diag").unwrap().to_vec1::<f32>()?;
    let hess_diag_est = tensors.get("hess_diag_est").unwrap().to_vec1::<f32>()?;
    let ada_v = tensors.get("ada_v").unwrap().to_vec1::<f32>()?;
    let ada_d = tensors.get("ada_d").unwrap().to_vec1::<f32>()?;
    let lbfgs_s = tensors.get("lbfgs_s").unwrap().to_vec2::<f32>()?;
    let lbfgs_y = tensors.get("lbfgs_y").unwrap().to_vec2::<f32>()?;
    let flat_s: Vec<f32> = lbfgs_s.iter().flatten().copied().collect();
    let flat_y: Vec<f32> = lbfgs_y.iter().flatten().copied().collect();

    let lm_jacobian = tensors.get("lm_jacobian").unwrap().to_vec2::<f32>()?;
    let flat_j: Vec<f32> = lm_jacobian.iter().flatten().copied().collect();
    let lm_residual = tensors.get("lm_residual").unwrap().to_vec1::<f32>()?;

    let gold_lbfgs = tensors
        .get("golden_lbfgs_theta")
        .unwrap()
        .to_vec1::<f32>()?;
    let gold_sophia_th = tensors
        .get("golden_sophia_theta")
        .unwrap()
        .to_vec1::<f32>()?;
    let gold_sophia_m = tensors.get("golden_sophia_m").unwrap().to_vec1::<f32>()?;
    let gold_sophia_h = tensors.get("golden_sophia_h").unwrap().to_vec1::<f32>()?;

    let gold_ada_th = tensors
        .get("golden_adahessian_theta")
        .unwrap()
        .to_vec1::<f32>()?;
    let gold_ada_m = tensors
        .get("golden_adahessian_m")
        .unwrap()
        .to_vec1::<f32>()?;
    let gold_ada_v = tensors
        .get("golden_adahessian_v")
        .unwrap()
        .to_vec1::<f32>()?;

    let gold_lm_th = tensors.get("golden_lm_theta").unwrap().to_vec1::<f32>()?;

    let d = theta.len();
    let m_hist = lbfgs_s.len();
    let n_res = lm_jacobian.len();

    // 1. L-BFGS CPU & Auto
    let cpu_lbfgs = t05::step_lbfgs_cpu(&theta, &flat_s, &flat_y, &grad, 0.1, m_hist, d);
    let auto_lbfgs = t05::step_lbfgs_auto(&kdev, &theta, &flat_s, &flat_y, &grad, 0.1, m_hist, d);
    for j in 0..d {
        assert!((cpu_lbfgs[j] - gold_lbfgs[j]).abs() < 1e-5);
        assert!((auto_lbfgs[j] - gold_lbfgs[j]).abs() < 1e-5);
    }

    // 2. Sophia-G CPU & Auto
    let (cpu_s_th, cpu_s_m, cpu_s_h) = t05::step_sophia_cpu(
        &theta,
        &grad,
        &momentum,
        &hess_diag,
        &hess_diag_est,
        0.01,
        0.96,
        0.99,
        0.01,
        0.04,
        1e-6,
    );
    let (auto_s_th, auto_s_m, auto_s_h) = t05::step_sophia_auto(
        &kdev,
        &theta,
        &grad,
        &momentum,
        &hess_diag,
        &hess_diag_est,
        0.01,
        0.96,
        0.99,
        0.01,
        0.04,
        1e-6,
    );
    for j in 0..d {
        assert!((cpu_s_th[j] - gold_sophia_th[j]).abs() < 1e-5);
        assert!((auto_s_th[j] - gold_sophia_th[j]).abs() < 1e-5);
        assert!((cpu_s_m[j] - gold_sophia_m[j]).abs() < 1e-5);
        assert!((auto_s_m[j] - gold_sophia_m[j]).abs() < 1e-5);
        assert!((cpu_s_h[j] - gold_sophia_h[j]).abs() < 1e-5);
        assert!((auto_s_h[j] - gold_sophia_h[j]).abs() < 1e-5);
    }

    // 3. AdaHessian CPU & Auto
    let (cpu_a_th, cpu_a_m, cpu_a_v) = t05::step_adahessian_cpu(
        &theta, &grad, &momentum, &ada_v, &ada_d, 0.01, 0.9, 0.999, 0.01, 1, 1e-4,
    );
    let (auto_a_th, auto_a_m, auto_a_v) = t05::step_adahessian_auto(
        &kdev, &theta, &grad, &momentum, &ada_v, &ada_d, 0.01, 0.9, 0.999, 0.01, 1, 1e-4,
    );
    for j in 0..d {
        assert!((cpu_a_th[j] - gold_ada_th[j]).abs() < 1e-5);
        assert!((auto_a_th[j] - gold_ada_th[j]).abs() < 1e-5);
        assert!((cpu_a_m[j] - gold_ada_m[j]).abs() < 1e-5);
        assert!((auto_a_m[j] - gold_ada_m[j]).abs() < 1e-5);
        assert!((cpu_a_v[j] - gold_ada_v[j]).abs() < 1e-5);
        assert!((auto_a_v[j] - gold_ada_v[j]).abs() < 1e-5);
    }

    // 4. Levenberg-Marquardt CPU & Auto
    let cpu_lm = t05::step_lm_cpu(&theta, &flat_j, &lm_residual, 0.1, n_res, d);
    let auto_lm = t05::step_lm_auto(&kdev, &theta, &flat_j, &lm_residual, 0.1, n_res, d);
    for j in 0..d {
        assert!((cpu_lm[j] - gold_lm_th[j]).abs() < 1e-5);
        assert!((auto_lm[j] - gold_lm_th[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_06_schedule_free_and_parameter_free_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t06_schedule_free_and_parameter_free_optimisers as t06;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let sf_x = tensors.get("sf_x").unwrap().to_vec1::<f32>()?;
    let sf_z = tensors.get("sf_z").unwrap().to_vec1::<f32>()?;
    let sf_y = tensors.get("sf_y").unwrap().to_vec1::<f32>()?;
    let sf_grad = tensors.get("sf_grad").unwrap().to_vec1::<f32>()?;
    let sf_m = tensors.get("sf_m").unwrap().to_vec1::<f32>()?;
    let sf_v = tensors.get("sf_v").unwrap().to_vec1::<f32>()?;

    let polyak_param = tensors.get("polyak_param").unwrap().to_vec1::<f32>()?;
    let polyak_grad = tensors.get("polyak_grad").unwrap().to_vec1::<f32>()?;
    let f_val = tensors.get("f_val").unwrap().to_vec1::<f32>()?[0];
    let f_opt = tensors.get("f_opt").unwrap().to_vec1::<f32>()?[0];

    let bb_param = tensors.get("bb_param").unwrap().to_vec1::<f32>()?;
    let bb_grad = tensors.get("bb_grad").unwrap().to_vec1::<f32>()?;
    let bb_s = tensors.get("bb_s").unwrap().to_vec1::<f32>()?;
    let bb_y = tensors.get("bb_y").unwrap().to_vec1::<f32>()?;

    let d_param = tensors.get("d_param").unwrap().to_vec1::<f32>()?;
    let d_grad = tensors.get("d_grad").unwrap().to_vec1::<f32>()?;
    let d_est = tensors.get("d_est").unwrap().to_vec1::<f32>()?[0];

    let gold_sf_x = tensors.get("golden_sf_x").unwrap().to_vec1::<f32>()?;
    let gold_sf_z = tensors.get("golden_sf_z").unwrap().to_vec1::<f32>()?;
    let gold_sf_y = tensors.get("golden_sf_y").unwrap().to_vec1::<f32>()?;
    let gold_polyak = tensors
        .get("golden_polyak_param")
        .unwrap()
        .to_vec1::<f32>()?;
    let gold_bb = tensors.get("golden_bb_param").unwrap().to_vec1::<f32>()?;
    let gold_dadapt = tensors
        .get("golden_d_adapt_param")
        .unwrap()
        .to_vec1::<f32>()?;

    let d = sf_x.len();

    // 1. Schedule-Free AdamW CPU & Auto
    let (cpu_x, cpu_z, cpu_y, _, _) = t06::step_schedule_free_cpu(
        &sf_x, &sf_z, &sf_y, &sf_grad, &sf_m, &sf_v, 0.01, 0.9, 0.999, 0.01, 1, 1e-8,
    );
    let (auto_x, auto_z, auto_y, _, _) = t06::step_schedule_free_auto(
        &kdev, &sf_x, &sf_z, &sf_y, &sf_grad, &sf_m, &sf_v, 0.01, 0.9, 0.999, 0.01, 1, 1e-8,
    );
    for j in 0..d {
        assert!((cpu_x[j] - gold_sf_x[j]).abs() < 1e-5);
        assert!((auto_x[j] - gold_sf_x[j]).abs() < 1e-5);
        assert!((cpu_z[j] - gold_sf_z[j]).abs() < 1e-5);
        assert!((auto_z[j] - gold_sf_z[j]).abs() < 1e-5);
        assert!((cpu_y[j] - gold_sf_y[j]).abs() < 1e-5);
        assert!((auto_y[j] - gold_sf_y[j]).abs() < 1e-5);
    }

    // 2. Polyak CPU & Auto
    let cpu_polyak = t06::step_polyak_cpu(&polyak_param, &polyak_grad, f_val, f_opt, 1e-8);
    let auto_polyak = t06::step_polyak_auto(&kdev, &polyak_param, &polyak_grad, f_val, f_opt, 1e-8);
    for j in 0..d {
        assert!((cpu_polyak[j] - gold_polyak[j]).abs() < 1e-5);
        assert!((auto_polyak[j] - gold_polyak[j]).abs() < 1e-5);
    }

    // 3. Barzilai-Borwein CPU & Auto
    let cpu_bb = t06::step_barzilai_borwein_cpu(&bb_param, &bb_grad, &bb_s, &bb_y, 1e-8);
    let auto_bb = t06::step_barzilai_borwein_auto(&kdev, &bb_param, &bb_grad, &bb_s, &bb_y, 1e-8);
    for j in 0..d {
        assert!((cpu_bb[j] - gold_bb[j]).abs() < 1e-5);
        assert!((auto_bb[j] - gold_bb[j]).abs() < 1e-5);
    }

    // 4. D-Adaptation CPU & Auto
    let (cpu_dadapt, _) = t06::step_d_adaptation_cpu(&d_param, &d_grad, d_est, 0.01, 1e-8);
    let (auto_dadapt, _) = t06::step_d_adaptation_auto(&kdev, &d_param, &d_grad, d_est, 0.01, 1e-8);
    for j in 0..d {
        assert!((cpu_dadapt[j] - gold_dadapt[j]).abs() < 1e-5);
        assert!((auto_dadapt[j] - gold_dadapt[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_07_learning_rate_schedulers_and_lr_finder_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t07_learning_rate_schedulers_and_lr_finder as t07;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param_cosine = tensors.get("param_cosine").unwrap().to_vec1::<f32>()?;
    let grad_cosine = tensors.get("grad_cosine").unwrap().to_vec1::<f32>()?;
    let gold_lr_cos = tensors.get("golden_lr_cosine").unwrap().to_vec1::<f32>()?[0];
    let gold_p_cos = tensors
        .get("golden_param_cosine")
        .unwrap()
        .to_vec1::<f32>()?;

    let param_wsd = tensors.get("param_wsd").unwrap().to_vec1::<f32>()?;
    let grad_wsd = tensors.get("grad_wsd").unwrap().to_vec1::<f32>()?;
    let gold_lr_wsd = tensors.get("golden_lr_wsd").unwrap().to_vec1::<f32>()?[0];
    let gold_p_wsd = tensors.get("golden_param_wsd").unwrap().to_vec1::<f32>()?;

    let param_onecycle = tensors.get("param_onecycle").unwrap().to_vec1::<f32>()?;
    let grad_onecycle = tensors.get("grad_onecycle").unwrap().to_vec1::<f32>()?;
    let gold_lr_oc = tensors
        .get("golden_lr_onecycle")
        .unwrap()
        .to_vec1::<f32>()?[0];
    let gold_p_oc = tensors
        .get("golden_param_onecycle")
        .unwrap()
        .to_vec1::<f32>()?;

    let param_autoclip = tensors.get("param_autoclip").unwrap().to_vec1::<f32>()?;
    let grad_autoclip = tensors.get("grad_autoclip").unwrap().to_vec1::<f32>()?;
    let norm_history = tensors.get("norm_history").unwrap().to_vec1::<f32>()?;
    let gold_clipped_g = tensors
        .get("golden_clipped_grad")
        .unwrap()
        .to_vec1::<f32>()?;
    let gold_p_ac = tensors
        .get("golden_param_autoclip")
        .unwrap()
        .to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let d = param_cosine.len();

    // 1. Cosine Annealing with Warmup CPU & Auto
    let cpu_lr_cos = t07::schedule_cosine_warmup_cpu(30, 100, 10, 1e-5, 0.01);
    let auto_lr_cos = t07::schedule_cosine_warmup_auto(&kdev, 30, 100, 10, 1e-5, 0.01);
    assert!((cpu_lr_cos - gold_lr_cos).abs() < 1e-5);
    assert!((auto_lr_cos - gold_lr_cos).abs() < 1e-5);
    let cpu_p_cos = t07::step_sgd_cpu(&param_cosine, &grad_cosine, cpu_lr_cos);
    let auto_p_cos = t07::step_sgd_auto(&kdev, &param_cosine, &grad_cosine, auto_lr_cos);
    for j in 0..d {
        assert!((cpu_p_cos[j] - gold_p_cos[j]).abs() < 1e-5);
        assert!((auto_p_cos[j] - gold_p_cos[j]).abs() < 1e-5);
    }

    // 2. WSD CPU & Auto
    let cpu_lr_wsd = t07::schedule_wsd_cpu(85, 100, 10, 20, 1e-4, 0.01);
    let auto_lr_wsd = t07::schedule_wsd_auto(&kdev, 85, 100, 10, 20, 1e-4, 0.01);
    assert!((cpu_lr_wsd - gold_lr_wsd).abs() < 1e-5);
    assert!((auto_lr_wsd - gold_lr_wsd).abs() < 1e-5);
    let cpu_p_wsd = t07::step_sgd_cpu(&param_wsd, &grad_wsd, cpu_lr_wsd);
    let auto_p_wsd = t07::step_sgd_auto(&kdev, &param_wsd, &grad_wsd, auto_lr_wsd);
    for j in 0..d {
        assert!((cpu_p_wsd[j] - gold_p_wsd[j]).abs() < 1e-5);
        assert!((auto_p_wsd[j] - gold_p_wsd[j]).abs() < 1e-5);
    }

    // 3. OneCycleLR CPU & Auto
    let cpu_lr_oc = t07::schedule_onecycle_cpu(40, 100, 30, 0.01, 25.0, 1000.0);
    let auto_lr_oc = t07::schedule_onecycle_auto(&kdev, 40, 100, 30, 0.01, 25.0, 1000.0);
    assert!((cpu_lr_oc - gold_lr_oc).abs() < 1e-5);
    assert!((auto_lr_oc - gold_lr_oc).abs() < 1e-5);
    let cpu_p_oc = t07::step_sgd_cpu(&param_onecycle, &grad_onecycle, cpu_lr_oc);
    let auto_p_oc = t07::step_sgd_auto(&kdev, &param_onecycle, &grad_onecycle, auto_lr_oc);
    for j in 0..d {
        assert!((cpu_p_oc[j] - gold_p_oc[j]).abs() < 1e-5);
        assert!((auto_p_oc[j] - gold_p_oc[j]).abs() < 1e-5);
    }

    // 4. AutoClip CPU & Auto
    let (cpu_p_ac, cpu_cg, _) = t07::step_autoclip_cpu(
        &param_autoclip,
        &grad_autoclip,
        &norm_history,
        90.0,
        0.01,
        1e-8,
    );
    let (auto_p_ac, auto_cg, _) = t07::step_autoclip_auto(
        &kdev,
        &param_autoclip,
        &grad_autoclip,
        &norm_history,
        90.0,
        0.01,
        1e-8,
    );
    for j in 0..d {
        assert!((cpu_cg[j] - gold_clipped_g[j]).abs() < 1e-5);
        assert!((auto_cg[j] - gold_clipped_g[j]).abs() < 1e-5);
        assert!((cpu_p_ac[j] - gold_p_ac[j]).abs() < 1e-5);
        assert!((auto_p_ac[j] - gold_p_ac[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t07::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_08_constrained_proximal_frank_wolfe_admm_and_mirror_descent_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t08_constrained_proximal_frank_wolfe_admm_and_mirror_descent as t08;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param_pgd = tensors.get("param_pgd").unwrap().to_vec1::<f32>()?;
    let grad_pgd = tensors.get("grad_pgd").unwrap().to_vec1::<f32>()?;
    let gold_p_pgd = tensors.get("golden_param_pgd").unwrap().to_vec1::<f32>()?;

    let param_fw = tensors.get("param_fw").unwrap().to_vec1::<f32>()?;
    let grad_fw = tensors.get("grad_fw").unwrap().to_vec1::<f32>()?;
    let gold_p_fw = tensors.get("golden_param_fw").unwrap().to_vec1::<f32>()?;

    let param_ista = tensors.get("param_ista").unwrap().to_vec1::<f32>()?;
    let grad_ista = tensors.get("grad_ista").unwrap().to_vec1::<f32>()?;
    let gold_p_ista = tensors.get("golden_param_ista").unwrap().to_vec1::<f32>()?;

    let param_md = tensors.get("param_md").unwrap().to_vec1::<f32>()?;
    let grad_md = tensors.get("grad_md").unwrap().to_vec1::<f32>()?;
    let gold_p_md = tensors.get("golden_param_md").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let d = param_pgd.len();

    // 1. PGD CPU & Auto
    let cpu_pgd = t08::step_pgd_cpu(&param_pgd, &grad_pgd, 0.1, 1.0, 1e-8);
    let auto_pgd = t08::step_pgd_auto(&kdev, &param_pgd, &grad_pgd, 0.1, 1.0, 1e-8);
    for j in 0..d {
        assert!((cpu_pgd[j] - gold_p_pgd[j]).abs() < 1e-5);
        assert!((auto_pgd[j] - gold_p_pgd[j]).abs() < 1e-5);
    }

    // 2. Frank-Wolfe CPU & Auto
    let cpu_fw = t08::step_frank_wolfe_cpu(&param_fw, &grad_fw, 1.0, 0.2);
    let auto_fw = t08::step_frank_wolfe_auto(&kdev, &param_fw, &grad_fw, 1.0, 0.2);
    for j in 0..d {
        assert!((cpu_fw[j] - gold_p_fw[j]).abs() < 1e-5);
        assert!((auto_fw[j] - gold_p_fw[j]).abs() < 1e-5);
    }

    // 3. ISTA CPU & Auto
    let cpu_ista = t08::step_ista_cpu(&param_ista, &grad_ista, 0.05, 0.1);
    let auto_ista = t08::step_ista_auto(&kdev, &param_ista, &grad_ista, 0.05, 0.1);
    for j in 0..d {
        assert!((cpu_ista[j] - gold_p_ista[j]).abs() < 1e-5);
        assert!((auto_ista[j] - gold_p_ista[j]).abs() < 1e-5);
    }

    // 4. Mirror Descent CPU & Auto
    let cpu_md = t08::step_mirror_descent_cpu(&param_md, &grad_md, 0.1, 1e-12);
    let auto_md = t08::step_mirror_descent_auto(&kdev, &param_md, &grad_md, 0.1, 1e-12);
    for j in 0..d {
        assert!((cpu_md[j] - gold_p_md[j]).abs() < 1e-5);
        assert!((auto_md[j] - gold_p_md[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t08::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_09_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery_cpu_and_auto() -> Result<()>
{
    use ai_kernel::c04_training_methods_and_optimisers::t09_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery as t09;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param = tensors.get("param").unwrap().to_vec1::<f32>()?;
    let g1 = tensors.get("g1").unwrap().to_vec1::<f32>()?;
    let g2 = tensors.get("g2").unwrap().to_vec1::<f32>()?;
    let losses = tensors.get("L").unwrap().to_vec1::<f32>()?;
    let z_star = tensors.get("z_star").unwrap().to_vec1::<f32>()?;
    let weights_tch = tensors.get("weights_tch").unwrap().to_vec1::<f32>()?;
    let weights_lin = tensors.get("weights_lin").unwrap().to_vec1::<f32>()?;

    let gold_alpha = tensors.get("golden_alpha_mgda").unwrap().to_vec1::<f32>()?[0];
    let gold_g_mgda = tensors.get("golden_g_mgda").unwrap().to_vec1::<f32>()?;
    let gold_p_mgda = tensors.get("golden_param_mgda").unwrap().to_vec1::<f32>()?;
    let gold_g_pc = tensors.get("golden_g_pc").unwrap().to_vec1::<f32>()?;
    let gold_p_pc = tensors.get("golden_param_pc").unwrap().to_vec1::<f32>()?;
    let gold_g_tch = tensors.get("golden_g_tch").unwrap().to_vec1::<f32>()?;
    let gold_p_tch = tensors.get("golden_param_tch").unwrap().to_vec1::<f32>()?;
    let gold_g_lin = tensors.get("golden_g_lin").unwrap().to_vec1::<f32>()?;
    let gold_p_lin = tensors.get("golden_param_lin").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let d = param.len();

    // 1. MGDA CPU & Auto
    let (cpu_p_mgda, cpu_g_mgda, cpu_alpha) =
        t09::step_mgda_2task_cpu(&param, &g1, &g2, 0.01, 1e-12);
    let (auto_p_mgda, auto_g_mgda, auto_alpha) =
        t09::step_mgda_2task_auto(&kdev, &param, &g1, &g2, 0.01, 1e-12);
    assert!((cpu_alpha - gold_alpha).abs() < 1e-5);
    assert!((auto_alpha - gold_alpha).abs() < 1e-5);
    for j in 0..d {
        assert!((cpu_g_mgda[j] - gold_g_mgda[j]).abs() < 1e-5);
        assert!((auto_g_mgda[j] - gold_g_mgda[j]).abs() < 1e-5);
        assert!((cpu_p_mgda[j] - gold_p_mgda[j]).abs() < 1e-5);
        assert!((auto_p_mgda[j] - gold_p_mgda[j]).abs() < 1e-5);
    }

    // 2. PCGrad CPU & Auto
    let (cpu_p_pc, cpu_g_pc) = t09::step_pcgrad_2task_cpu(&param, &g1, &g2, 0.01, 1e-12);
    let (auto_p_pc, auto_g_pc) = t09::step_pcgrad_2task_auto(&kdev, &param, &g1, &g2, 0.01, 1e-12);
    for j in 0..d {
        assert!((cpu_g_pc[j] - gold_g_pc[j]).abs() < 1e-5);
        assert!((auto_g_pc[j] - gold_g_pc[j]).abs() < 1e-5);
        assert!((cpu_p_pc[j] - gold_p_pc[j]).abs() < 1e-5);
        assert!((auto_p_pc[j] - gold_p_pc[j]).abs() < 1e-5);
    }

    // 3. Tchebycheff CPU & Auto
    let (cpu_p_tch, cpu_g_tch) =
        t09::step_tchebycheff_cpu(&param, &g1, &g2, &losses, &z_star, &weights_tch, 0.05, 0.01);
    let (auto_p_tch, auto_g_tch) = t09::step_tchebycheff_auto(
        &kdev,
        &param,
        &g1,
        &g2,
        &losses,
        &z_star,
        &weights_tch,
        0.05,
        0.01,
    );
    for j in 0..d {
        assert!((cpu_g_tch[j] - gold_g_tch[j]).abs() < 1e-5);
        assert!((auto_g_tch[j] - gold_g_tch[j]).abs() < 1e-5);
        assert!((cpu_p_tch[j] - gold_p_tch[j]).abs() < 1e-5);
        assert!((auto_p_tch[j] - gold_p_tch[j]).abs() < 1e-5);
    }

    // 4. Linear Scalarization CPU & Auto
    let (cpu_p_lin, cpu_g_lin) =
        t09::step_linear_scalarization_cpu(&param, &g1, &g2, &weights_lin, 0.01);
    let (auto_p_lin, auto_g_lin) =
        t09::step_linear_scalarization_auto(&kdev, &param, &g1, &g2, &weights_lin, 0.01);
    for j in 0..d {
        assert!((cpu_g_lin[j] - gold_g_lin[j]).abs() < 1e-5);
        assert!((auto_g_lin[j] - gold_g_lin[j]).abs() < 1e-5);
        assert!((cpu_p_lin[j] - gold_p_lin[j]).abs() < 1e-5);
        assert!((auto_p_lin[j] - gold_p_lin[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t09::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_10_coordinate_alternating_projections_and_majorization_minimization_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t10_coordinate_alternating_projections_and_majorization_minimization as t10;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let x0 = tensors.get("x0").unwrap().to_vec1::<f32>()?;
    let normal = tensors.get("normal").unwrap().to_vec1::<f32>()?;
    let offset = tensors.get("offset").unwrap().to_vec1::<f32>()?[0];

    let param_bcd = tensors.get("param_bcd").unwrap().to_vec1::<f32>()?;
    let b_bcd = tensors.get("b_bcd").unwrap().to_vec1::<f32>()?;
    let a_diag = tensors.get("A_diag").unwrap().to_vec1::<f32>()?;

    let param_irls = tensors.get("param_irls").unwrap().to_vec1::<f32>()?;
    let target_irls = tensors.get("target_irls").unwrap().to_vec1::<f32>()?;

    let param_cccp = tensors.get("param_cccp").unwrap().to_vec1::<f32>()?;

    let gold_dykstra_x = tensors.get("golden_dykstra_x").unwrap().to_vec1::<f32>()?;
    let gold_dykstra_y = tensors.get("golden_dykstra_y").unwrap().to_vec1::<f32>()?;
    let gold_dykstra_p = tensors.get("golden_dykstra_p").unwrap().to_vec1::<f32>()?;
    let gold_dykstra_q = tensors.get("golden_dykstra_q").unwrap().to_vec1::<f32>()?;

    let gold_param_bcd = tensors.get("golden_param_bcd").unwrap().to_vec1::<f32>()?;
    let gold_param_irls = tensors.get("golden_param_irls").unwrap().to_vec1::<f32>()?;
    let gold_param_cccp = tensors.get("golden_param_cccp").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let d = x0.len();

    // 1. Dykstra CPU & Auto
    let p0 = vec![0.0f32; d];
    let q0 = vec![0.0f32; d];
    let (cpu_x, cpu_y, cpu_p, cpu_q) =
        t10::step_dykstra_cpu(&x0, &p0, &q0, &normal, offset, 0.0, 2.0);
    let (auto_x, auto_y, auto_p, auto_q) =
        t10::step_dykstra_auto(&kdev, &x0, &p0, &q0, &normal, offset, 0.0, 2.0);
    for j in 0..d {
        assert!((cpu_x[j] - gold_dykstra_x[j]).abs() < 1e-5);
        assert!((auto_x[j] - gold_dykstra_x[j]).abs() < 1e-5);
        assert!((cpu_y[j] - gold_dykstra_y[j]).abs() < 1e-5);
        assert!((auto_y[j] - gold_dykstra_y[j]).abs() < 1e-5);
        assert!((cpu_p[j] - gold_dykstra_p[j]).abs() < 1e-5);
        assert!((auto_p[j] - gold_dykstra_p[j]).abs() < 1e-5);
        assert!((cpu_q[j] - gold_dykstra_q[j]).abs() < 1e-5);
        assert!((auto_q[j] - gold_dykstra_q[j]).abs() < 1e-5);
    }

    // 2. BCD CPU & Auto
    let cpu_bcd = t10::step_bcd_cpu(&param_bcd, &b_bcd, &a_diag, 0.2);
    let auto_bcd = t10::step_bcd_auto(&kdev, &param_bcd, &b_bcd, &a_diag, 0.2);
    for j in 0..d {
        assert!((cpu_bcd[j] - gold_param_bcd[j]).abs() < 1e-5);
        assert!((auto_bcd[j] - gold_param_bcd[j]).abs() < 1e-5);
    }

    // 3. IRLS CPU & Auto
    let cpu_irls = t10::step_irls_cpu(&param_irls, &target_irls, 0.5, 1e-4);
    let auto_irls = t10::step_irls_auto(&kdev, &param_irls, &target_irls, 0.5, 1e-4);
    for j in 0..d {
        assert!((cpu_irls[j] - gold_param_irls[j]).abs() < 1e-5);
        assert!((auto_irls[j] - gold_param_irls[j]).abs() < 1e-5);
    }

    // 4. CCCP CPU & Auto
    let cpu_cccp = t10::step_cccp_cpu(&param_cccp);
    let auto_cccp = t10::step_cccp_auto(&kdev, &param_cccp);
    for j in 0..d {
        assert!((cpu_cccp[j] - gold_param_cccp[j]).abs() < 1e-5);
        assert!((auto_cccp[j] - gold_param_cccp[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t10::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_11_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t11_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn as t11;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let log_alpha = tensors.get("log_alpha").unwrap().to_vec2::<f32>()?;
    let flat_log_alpha: Vec<f32> = log_alpha.iter().flatten().copied().collect();

    let candidates = tensors.get("candidates").unwrap().to_vec2::<f32>()?;
    let flat_candidates: Vec<f32> = candidates.iter().flatten().copied().collect();
    let scores = tensors.get("scores").unwrap().to_vec1::<f32>()?;

    let delta_e = tensors.get("delta_e").unwrap().to_vec1::<f32>()?;
    let sa_temp = tensors.get("sa_temp").unwrap().to_vec1::<f32>()?[0];

    let pt_e1 = tensors.get("pt_e1").unwrap().to_vec1::<f32>()?[0];
    let pt_e2 = tensors.get("pt_e2").unwrap().to_vec1::<f32>()?[0];
    let pt_t1 = tensors.get("pt_t1").unwrap().to_vec1::<f32>()?[0];
    let pt_t2 = tensors.get("pt_t2").unwrap().to_vec1::<f32>()?[0];

    let gold_sinkhorn = tensors
        .get("golden_sinkhorn_out")
        .unwrap()
        .to_vec2::<f32>()?;
    let flat_gold_sinkhorn: Vec<f32> = gold_sinkhorn.iter().flatten().copied().collect();

    let gold_cem_mean = tensors.get("golden_cem_mean").unwrap().to_vec1::<f32>()?;
    let gold_cem_std = tensors.get("golden_cem_std").unwrap().to_vec1::<f32>()?;
    let gold_sa_probs = tensors.get("golden_sa_probs").unwrap().to_vec1::<f32>()?;
    let gold_pt_prob = tensors
        .get("golden_pt_swap_prob")
        .unwrap()
        .to_vec1::<f32>()?[0];

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    // 1. Sinkhorn CPU & Auto
    let cpu_sinkhorn = t11::run_sinkhorn_cpu(&flat_log_alpha, 4, 4, 10, 0.5);
    let auto_sinkhorn = t11::run_sinkhorn_auto(&kdev, &flat_log_alpha, 4, 4, 10, 0.5);
    for j in 0..flat_gold_sinkhorn.len() {
        assert!((cpu_sinkhorn[j] - flat_gold_sinkhorn[j]).abs() < 1e-5);
        assert!((auto_sinkhorn[j] - flat_gold_sinkhorn[j]).abs() < 1e-5);
    }

    // 2. CEM CPU & Auto
    let (cpu_mean, cpu_std) = t11::step_cem_cpu(&flat_candidates, &scores, 20, 4, 0.25);
    let (auto_mean, auto_std) = t11::step_cem_auto(&kdev, &flat_candidates, &scores, 20, 4, 0.25);
    for j in 0..4 {
        assert!((cpu_mean[j] - gold_cem_mean[j]).abs() < 1e-5);
        assert!((auto_mean[j] - gold_cem_mean[j]).abs() < 1e-5);
        assert!((cpu_std[j] - gold_cem_std[j]).abs() < 1e-5);
        assert!((auto_std[j] - gold_cem_std[j]).abs() < 1e-5);
    }

    // 3. Simulated Annealing CPU & Auto
    let cpu_sa = t11::sa_acceptance_prob_cpu(&delta_e, sa_temp);
    let auto_sa = t11::sa_acceptance_prob_auto(&kdev, &delta_e, sa_temp);
    for j in 0..delta_e.len() {
        assert!((cpu_sa[j] - gold_sa_probs[j]).abs() < 1e-5);
        assert!((auto_sa[j] - gold_sa_probs[j]).abs() < 1e-5);
    }

    // 4. Parallel Tempering CPU & Auto
    let cpu_pt = t11::parallel_tempering_swap_prob_cpu(pt_e1, pt_e2, pt_t1, pt_t2);
    let auto_pt = t11::parallel_tempering_swap_prob_auto(&kdev, pt_e1, pt_e2, pt_t1, pt_t2);
    assert!((cpu_pt - gold_pt_prob).abs() < 1e-5);
    assert!((auto_pt - gold_pt_prob).abs() < 1e-5);

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t11::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_12_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t12_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed as t12;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let sha_losses = tensors.get("sha_losses").unwrap().to_vec1::<f32>()?;
    let golden_sha_idx = tensors.get("golden_sha_idx").unwrap().to_vec1::<f32>()?;
    let golden_sha_losses = tensors.get("golden_sha_losses").unwrap().to_vec1::<f32>()?;

    let golden_hb_schedule = tensors
        .get("golden_hb_schedule")
        .unwrap()
        .to_vec2::<f32>()?;
    let flat_golden_hb_schedule: Vec<f32> = golden_hb_schedule.iter().flatten().copied().collect();

    let x_query = tensors.get("x_query").unwrap().to_vec1::<f32>()?;
    let good_samples = tensors.get("good_samples").unwrap().to_vec1::<f32>()?;
    let bad_samples = tensors.get("bad_samples").unwrap().to_vec1::<f32>()?;
    let golden_tpe_log_ratio = tensors
        .get("golden_tpe_log_ratio")
        .unwrap()
        .to_vec1::<f32>()?;

    let rung_losses = tensors.get("rung_losses").unwrap().to_vec1::<f32>()?;
    let candidate_losses = tensors.get("candidate_losses").unwrap().to_vec1::<f32>()?;
    let golden_asha_promoted = tensors
        .get("golden_asha_promoted")
        .unwrap()
        .to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    // 1. SHA CPU & Auto
    let (cpu_idx, cpu_losses) = t12::compute_sha_top_k_cpu(&sha_losses, 3);
    let (auto_idx, auto_losses) = t12::compute_sha_top_k_auto(&kdev, &sha_losses, 3);
    for j in 0..golden_sha_idx.len() {
        assert!((cpu_idx[j] - golden_sha_idx[j]).abs() < 1e-5);
        assert!((auto_idx[j] - golden_sha_idx[j]).abs() < 1e-5);
        assert!((cpu_losses[j] - golden_sha_losses[j]).abs() < 1e-5);
        assert!((auto_losses[j] - golden_sha_losses[j]).abs() < 1e-5);
    }

    // 2. Hyperband CPU & Auto
    let cpu_hb = t12::compute_hyperband_schedule_cpu(81, 3);
    let auto_hb = t12::compute_hyperband_schedule_auto(&kdev, 81, 3);
    for j in 0..flat_golden_hb_schedule.len() {
        assert!((cpu_hb[j] - flat_golden_hb_schedule[j]).abs() < 1e-5);
        assert!((auto_hb[j] - flat_golden_hb_schedule[j]).abs() < 1e-5);
    }

    // 3. TPE CPU & Auto
    let cpu_tpe = t12::compute_tpe_log_ratio_cpu(&x_query, &good_samples, &bad_samples, 0.5);
    let auto_tpe =
        t12::compute_tpe_log_ratio_auto(&kdev, &x_query, &good_samples, &bad_samples, 0.5);
    for j in 0..golden_tpe_log_ratio.len() {
        assert!((cpu_tpe[j] - golden_tpe_log_ratio[j]).abs() < 1e-5);
        assert!((auto_tpe[j] - golden_tpe_log_ratio[j]).abs() < 1e-5);
    }

    // 4. ASHA CPU & Auto
    let cpu_p0 = t12::check_asha_promotion_cpu(&rung_losses, candidate_losses[0], 3);
    let cpu_p1 = t12::check_asha_promotion_cpu(&rung_losses, candidate_losses[1], 3);
    let auto_p0 = t12::check_asha_promotion_auto(&kdev, &rung_losses, candidate_losses[0], 3);
    let auto_p1 = t12::check_asha_promotion_auto(&kdev, &rung_losses, candidate_losses[1], 3);
    assert!((cpu_p0 - golden_asha_promoted[0]).abs() < 1e-5);
    assert!((cpu_p1 - golden_asha_promoted[1]).abs() < 1e-5);
    assert!((auto_p0 - golden_asha_promoted[0]).abs() < 1e-5);
    assert!((auto_p1 - golden_asha_promoted[1]).abs() < 1e-5);

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t12::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_13_sharpness_aware_swad_and_weight_averaging_dynamics_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t13_sharpness_aware_swad_and_weight_averaging_dynamics as t13;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param_sam = tensors.get("param_sam").unwrap().to_vec1::<f32>()?;
    let grad_at_w = tensors.get("grad_at_w").unwrap().to_vec1::<f32>()?;
    let grad_at_pert = tensors.get("grad_at_pert").unwrap().to_vec1::<f32>()?;
    let golden_param_sam = tensors.get("golden_param_sam").unwrap().to_vec1::<f32>()?;
    let golden_epsilon_sam = tensors
        .get("golden_epsilon_sam")
        .unwrap()
        .to_vec1::<f32>()?;

    let slow_p = tensors.get("slow_p").unwrap().to_vec1::<f32>()?;
    let fast_p = tensors.get("fast_p").unwrap().to_vec1::<f32>()?;
    let golden_slow_p = tensors.get("golden_slow_p").unwrap().to_vec1::<f32>()?;
    let golden_fast_p = tensors.get("golden_fast_p").unwrap().to_vec1::<f32>()?;

    let swa_avg = tensors.get("swa_avg").unwrap().to_vec1::<f32>()?;
    let ema_avg = tensors.get("ema_avg").unwrap().to_vec1::<f32>()?;
    let curr_param = tensors.get("curr_param").unwrap().to_vec1::<f32>()?;
    let golden_swa = tensors.get("golden_swa").unwrap().to_vec1::<f32>()?;
    let golden_ema = tensors.get("golden_ema").unwrap().to_vec1::<f32>()?;

    let soup_models = tensors
        .get("soup_models")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_soup = tensors.get("golden_soup").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    // 1. SAM CPU & Auto
    let (cpu_p, cpu_eps) = t13::step_sam_cpu(
        &param_sam,
        &grad_at_w,
        &grad_at_pert,
        0.05,
        0.01,
        0.001,
        1e-12,
    );
    let (auto_p, auto_eps) = t13::step_sam_auto(
        &kdev,
        &param_sam,
        &grad_at_w,
        &grad_at_pert,
        0.05,
        0.01,
        0.001,
        1e-12,
    );
    for j in 0..golden_param_sam.len() {
        assert!((cpu_p[j] - golden_param_sam[j]).abs() < 1e-5);
        assert!((auto_p[j] - golden_param_sam[j]).abs() < 1e-5);
        assert!((cpu_eps[j] - golden_epsilon_sam[j]).abs() < 1e-5);
        assert!((auto_eps[j] - golden_epsilon_sam[j]).abs() < 1e-5);
    }

    // 2. Lookahead CPU & Auto
    let (cpu_slow, cpu_fast) = t13::step_lookahead_cpu(&slow_p, &fast_p, 0.5);
    let (auto_slow, auto_fast) = t13::step_lookahead_auto(&kdev, &slow_p, &fast_p, 0.5);
    for j in 0..golden_slow_p.len() {
        assert!((cpu_slow[j] - golden_slow_p[j]).abs() < 1e-5);
        assert!((auto_slow[j] - golden_slow_p[j]).abs() < 1e-5);
        assert!((cpu_fast[j] - golden_fast_p[j]).abs() < 1e-5);
        assert!((auto_fast[j] - golden_fast_p[j]).abs() < 1e-5);
    }

    // 3. SWA & EMA CPU & Auto
    let (cpu_swa, cpu_ema) = t13::step_swa_ema_cpu(&swa_avg, &ema_avg, &curr_param, 4, 0.99);
    let (auto_swa, auto_ema) =
        t13::step_swa_ema_auto(&kdev, &swa_avg, &ema_avg, &curr_param, 4, 0.99);
    for j in 0..golden_swa.len() {
        assert!((cpu_swa[j] - golden_swa[j]).abs() < 1e-5);
        assert!((auto_swa[j] - golden_swa[j]).abs() < 1e-5);
        assert!((cpu_ema[j] - golden_ema[j]).abs() < 1e-5);
        assert!((auto_ema[j] - golden_ema[j]).abs() < 1e-5);
    }

    // 4. Model Soup CPU & Auto
    let cpu_soup = t13::uniform_model_soup_cpu(&soup_models, 3, 4);
    let auto_soup = t13::uniform_model_soup_auto(&kdev, &soup_models, 3, 4);
    for j in 0..golden_soup.len() {
        assert!((cpu_soup[j] - golden_soup[j]).abs() < 1e-5);
        assert!((auto_soup[j] - golden_soup[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t13::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_14_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t14_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa as t14;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let matrix_a = tensors
        .get("matrix_A")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let state_x = tensors.get("state_x").unwrap().to_vec1::<f32>()?;
    let state_y = tensors.get("state_y").unwrap().to_vec1::<f32>()?;
    let state_x_prev = tensors.get("state_x_prev").unwrap().to_vec1::<f32>()?;
    let state_y_prev = tensors.get("state_y_prev").unwrap().to_vec1::<f32>()?;

    let golden_x_eg = tensors.get("golden_x_eg").unwrap().to_vec1::<f32>()?;
    let golden_y_eg = tensors.get("golden_y_eg").unwrap().to_vec1::<f32>()?;
    let golden_x_ogda = tensors.get("golden_x_ogda").unwrap().to_vec1::<f32>()?;
    let golden_y_ogda = tensors.get("golden_y_ogda").unwrap().to_vec1::<f32>()?;
    let golden_x_sga = tensors.get("golden_x_sga").unwrap().to_vec1::<f32>()?;
    let golden_y_sga = tensors.get("golden_y_sga").unwrap().to_vec1::<f32>()?;

    let x_best = tensors.get("x_best").unwrap().to_vec1::<f32>()?;
    let x_second = tensors.get("x_second").unwrap().to_vec1::<f32>()?;
    let x_worst = tensors.get("x_worst").unwrap().to_vec1::<f32>()?;
    let golden_x_reflect = tensors.get("golden_x_reflect").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    // 1. Extragradient CPU & Auto
    let (cpu_x_eg, cpu_y_eg) = t14::step_extragradient_cpu(&state_x, &state_y, &matrix_a, 0.1, 0.1);
    let (auto_x_eg, auto_y_eg) =
        t14::step_extragradient_auto(&kdev, &state_x, &state_y, &matrix_a, 0.1, 0.1);
    for j in 0..golden_x_eg.len() {
        assert!((cpu_x_eg[j] - golden_x_eg[j]).abs() < 1e-5);
        assert!((auto_x_eg[j] - golden_x_eg[j]).abs() < 1e-5);
        assert!((cpu_y_eg[j] - golden_y_eg[j]).abs() < 1e-5);
        assert!((auto_y_eg[j] - golden_y_eg[j]).abs() < 1e-5);
    }

    // 2. OGDA CPU & Auto
    let (cpu_x_ogda, cpu_y_ogda) = t14::step_ogda_cpu(
        &state_x,
        &state_y,
        &state_x_prev,
        &state_y_prev,
        &matrix_a,
        0.1,
        0.1,
    );
    let (auto_x_ogda, auto_y_ogda) = t14::step_ogda_auto(
        &kdev,
        &state_x,
        &state_y,
        &state_x_prev,
        &state_y_prev,
        &matrix_a,
        0.1,
        0.1,
    );
    for j in 0..golden_x_ogda.len() {
        assert!((cpu_x_ogda[j] - golden_x_ogda[j]).abs() < 1e-5);
        assert!((auto_x_ogda[j] - golden_x_ogda[j]).abs() < 1e-5);
        assert!((cpu_y_ogda[j] - golden_y_ogda[j]).abs() < 1e-5);
        assert!((auto_y_ogda[j] - golden_y_ogda[j]).abs() < 1e-5);
    }

    // 3. SGA CPU & Auto
    let (cpu_x_sga, cpu_y_sga) = t14::step_sga_cpu(&state_x, &state_y, &matrix_a, 0.5, 0.1, 0.1);
    let (auto_x_sga, auto_y_sga) =
        t14::step_sga_auto(&kdev, &state_x, &state_y, &matrix_a, 0.5, 0.1, 0.1);
    for j in 0..golden_x_sga.len() {
        assert!((cpu_x_sga[j] - golden_x_sga[j]).abs() < 1e-5);
        assert!((auto_x_sga[j] - golden_x_sga[j]).abs() < 1e-5);
        assert!((cpu_y_sga[j] - golden_y_sga[j]).abs() < 1e-5);
        assert!((auto_y_sga[j] - golden_y_sga[j]).abs() < 1e-5);
    }

    // 4. Nelder-Mead Reflection CPU & Auto
    let cpu_refl = t14::nelder_mead_reflect_cpu(&x_best, &x_second, &x_worst, 1.0);
    let auto_refl = t14::nelder_mead_reflect_auto(&kdev, &x_best, &x_second, &x_worst, 1.0);
    for j in 0..golden_x_reflect.len() {
        assert!((cpu_refl[j] - golden_x_reflect[j]).abs() < 1e-5);
        assert!((auto_refl[j] - golden_x_reflect[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t14::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_15_grokking_phase_transitions_edge_of_stability_and_implicit_bias_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t15_grokking_phase_transitions_edge_of_stability_and_implicit_bias as t15;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let g_raw = tensors.get("g_raw").unwrap().to_vec1::<f32>()?;
    let mu_prev = tensors.get("mu_prev").unwrap().to_vec1::<f32>()?;
    let golden_mu = tensors.get("golden_mu").unwrap().to_vec1::<f32>()?;
    let golden_g_grokfast = tensors.get("golden_g_grokfast").unwrap().to_vec1::<f32>()?;

    let x_eos = tensors.get("x_eos").unwrap().to_vec1::<f32>()?;
    let h_eos = tensors.get("h_eos").unwrap().to_vec1::<f32>()?;
    let gamma_eos = tensors.get("gamma_eos").unwrap().to_vec1::<f32>()?;
    let golden_x_eos = tensors.get("golden_x_eos").unwrap().to_vec1::<f32>()?;

    let w_sep = tensors.get("w_sep").unwrap().to_vec1::<f32>()?;
    let x_sep = tensors
        .get("x_sep")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let y_sep = tensors.get("y_sep").unwrap().to_vec1::<f32>()?;
    let golden_w_sep = tensors.get("golden_w_sep").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. Grokfast CPU & Auto
    let (cpu_gf, cpu_mu) = t15::step_grokfast_cpu(&g_raw, &mu_prev, 0.8, 2.0);
    let (auto_gf, auto_mu) = t15::step_grokfast_auto(&kdev, &g_raw, &mu_prev, 0.8, 2.0);
    for j in 0..golden_mu.len() {
        assert!((cpu_mu[j] - golden_mu[j]).abs() < 1e-5);
        assert!((auto_mu[j] - golden_mu[j]).abs() < 1e-5);
        assert!((cpu_gf[j] - golden_g_grokfast[j]).abs() < 1e-5);
        assert!((auto_gf[j] - golden_g_grokfast[j]).abs() < 1e-5);
    }

    // 2. EOS CPU & Auto
    let cpu_eos = t15::step_edge_of_stability_cpu(&x_eos, &h_eos, &gamma_eos, 0.1);
    let auto_eos = t15::step_edge_of_stability_auto(&kdev, &x_eos, &h_eos, &gamma_eos, 0.1);
    for j in 0..golden_x_eos.len() {
        assert!((cpu_eos[j] - golden_x_eos[j]).abs() < 1e-5);
        assert!((auto_eos[j] - golden_x_eos[j]).abs() < 1e-5);
    }

    // 3. Implicit Bias CPU & Auto
    let cpu_w_sep = t15::step_implicit_bias_cpu(&w_sep, &x_sep, &y_sep, 0.2);
    let auto_w_sep = t15::step_implicit_bias_auto(&kdev, &w_sep, &x_sep, &y_sep, 0.2);
    for j in 0..golden_w_sep.len() {
        assert!((cpu_w_sep[j] - golden_w_sep[j]).abs() < 1e-5);
        assert!((auto_w_sep[j] - golden_w_sep[j]).abs() < 1e-5);
    }

    // 4. Linear Anchor CPU
    let cpu_anchor_out = t15::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_16_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t16_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity as t16;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let z_emb = tensors
        .get("z_emb")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mrl_w = tensors
        .get("mrl_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mrl_b = tensors
        .get("mrl_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_logits_d2 = tensors
        .get("golden_logits_d2")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_logits_d4 = tensors
        .get("golden_logits_d4")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_logits_d8 = tensors
        .get("golden_logits_d8")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let p_probs = tensors
        .get("p_probs")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let y_targets = tensors
        .get("y_targets")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_el2n = tensors.get("golden_el2n").unwrap().to_vec1::<f32>()?;
    let loss_model = tensors.get("loss_model").unwrap().to_vec1::<f32>()?;
    let loss_ref = tensors.get("loss_ref").unwrap().to_vec1::<f32>()?;
    let golden_rho_loss = tensors.get("golden_rho_loss").unwrap().to_vec1::<f32>()?;

    let activations = tensors
        .get("activations")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_mean_act = tensors.get("mean_act").unwrap().to_vec1::<f32>()?;
    let golden_dormant_mask = tensors.get("dormant_mask").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. MRL CPU & Auto
    let cpu_d2 = t16::mrl_slice_forward_cpu(&z_emb, &mrl_w, &mrl_b, 4, 8, 4, 2);
    let auto_d2 = t16::mrl_slice_forward_auto(&kdev, &z_emb, &mrl_w, &mrl_b, 4, 8, 4, 2);
    for j in 0..golden_logits_d2.len() {
        assert!((cpu_d2[j] - golden_logits_d2[j]).abs() < 1e-5);
        assert!((auto_d2[j] - golden_logits_d2[j]).abs() < 1e-5);
    }

    let cpu_d4 = t16::mrl_slice_forward_cpu(&z_emb, &mrl_w, &mrl_b, 4, 8, 4, 4);
    let auto_d4 = t16::mrl_slice_forward_auto(&kdev, &z_emb, &mrl_w, &mrl_b, 4, 8, 4, 4);
    for j in 0..golden_logits_d4.len() {
        assert!((cpu_d4[j] - golden_logits_d4[j]).abs() < 1e-5);
        assert!((auto_d4[j] - golden_logits_d4[j]).abs() < 1e-5);
    }

    let cpu_d8 = t16::mrl_slice_forward_cpu(&z_emb, &mrl_w, &mrl_b, 4, 8, 4, 8);
    let auto_d8 = t16::mrl_slice_forward_auto(&kdev, &z_emb, &mrl_w, &mrl_b, 4, 8, 4, 8);
    for j in 0..golden_logits_d8.len() {
        assert!((cpu_d8[j] - golden_logits_d8[j]).abs() < 1e-5);
        assert!((auto_d8[j] - golden_logits_d8[j]).abs() < 1e-5);
    }

    // 2. EL2N CPU & Auto
    let cpu_el2n = t16::compute_el2n_cpu(&p_probs, &y_targets, 4, 3);
    let auto_el2n = t16::compute_el2n_auto(&kdev, &p_probs, &y_targets, 4, 3);
    for j in 0..golden_el2n.len() {
        assert!((cpu_el2n[j] - golden_el2n[j]).abs() < 1e-5);
        assert!((auto_el2n[j] - golden_el2n[j]).abs() < 1e-5);
    }

    // 3. RHO-Loss CPU & Auto
    let cpu_rho = t16::compute_rho_loss_cpu(&loss_model, &loss_ref);
    let auto_rho = t16::compute_rho_loss_auto(&kdev, &loss_model, &loss_ref);
    for j in 0..golden_rho_loss.len() {
        assert!((cpu_rho[j] - golden_rho_loss[j]).abs() < 1e-5);
        assert!((auto_rho[j] - golden_rho_loss[j]).abs() < 1e-5);
    }

    // 4. ReDo Dormant Neurons CPU & Auto
    let (cpu_mean, cpu_mask) = t16::detect_dormant_neurons_cpu(&activations, 4, 6, 0.15);
    let (auto_mean, auto_mask) = t16::detect_dormant_neurons_auto(&kdev, &activations, 4, 6, 0.15);
    for j in 0..golden_mean_act.len() {
        assert!((cpu_mean[j] - golden_mean_act[j]).abs() < 1e-5);
        assert!((auto_mean[j] - golden_mean_act[j]).abs() < 1e-5);
        assert!((cpu_mask[j] - golden_dormant_mask[j]).abs() < 1e-5);
        assert!((auto_mask[j] - golden_dormant_mask[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t16::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_17_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t17_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting as t17;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let w_hd = tensors
        .get("w_hd")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let g_curr = tensors
        .get("g_curr")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let g_prev = tensors
        .get("g_prev")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let lr_init = tensors
        .get("lr_init")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let hyper_lr = tensors
        .get("hyper_lr")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_lr_next = tensors
        .get("golden_lr_next")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_w_hd = tensors
        .get("golden_w_hd")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let per_sample_grads = tensors
        .get("per_sample_grads")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let val_grad = tensors
        .get("val_grad")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_sample_weights = tensors
        .get("golden_sample_weights")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_reweighted_grad = tensors
        .get("golden_reweighted_grad")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let param_learned = tensors
        .get("param_learned")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let grad_learned = tensors
        .get("grad_learned")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let momentum_learned = tensors
        .get("momentum_learned")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mlp_w1 = tensors
        .get("mlp_w1")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mlp_b1 = tensors
        .get("mlp_b1")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mlp_w2 = tensors
        .get("mlp_w2")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mlp_b2 = tensors
        .get("mlp_b2")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_delta_w = tensors
        .get("golden_delta_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_param_learned = tensors
        .get("golden_param_learned")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_momentum_learned = tensors
        .get("golden_momentum_learned")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. Hypergradient Descent CPU & Auto
    let (cpu_w, cpu_lr) =
        t17::step_hypergradient_descent_cpu(&w_hd, &g_curr, &g_prev, lr_init, hyper_lr);
    let (auto_w, auto_lr) =
        t17::step_hypergradient_descent_auto(&kdev, &w_hd, &g_curr, &g_prev, lr_init, hyper_lr);
    assert!((cpu_lr - golden_lr_next).abs() < 1e-5);
    assert!((auto_lr - golden_lr_next).abs() < 1e-5);
    for j in 0..golden_w_hd.len() {
        assert!((cpu_w[j] - golden_w_hd[j]).abs() < 1e-5);
        assert!((auto_w[j] - golden_w_hd[j]).abs() < 1e-5);
    }

    // 2. Meta-Sample Reweighting CPU & Auto
    let (cpu_weights, cpu_rew_g) =
        t17::meta_sample_reweight_cpu(&per_sample_grads, &val_grad, 4, 8, 1e-8);
    let (auto_weights, auto_rew_g) =
        t17::meta_sample_reweight_auto(&kdev, &per_sample_grads, &val_grad, 4, 8, 1e-8);
    for j in 0..golden_sample_weights.len() {
        assert!((cpu_weights[j] - golden_sample_weights[j]).abs() < 1e-5);
        assert!((auto_weights[j] - golden_sample_weights[j]).abs() < 1e-5);
    }
    for j in 0..golden_reweighted_grad.len() {
        assert!((cpu_rew_g[j] - golden_reweighted_grad[j]).abs() < 1e-5);
        assert!((auto_rew_g[j] - golden_reweighted_grad[j]).abs() < 1e-5);
    }

    // 3. Learned Optimizer MLP CPU & Auto
    let (cpu_p, cpu_m, cpu_dw) = t17::step_learned_optimizer_cpu(
        &param_learned,
        &grad_learned,
        &momentum_learned,
        &mlp_w1,
        &mlp_b1,
        &mlp_w2,
        &mlp_b2,
    );
    let (auto_p, auto_m, auto_dw) = t17::step_learned_optimizer_auto(
        &kdev,
        &param_learned,
        &grad_learned,
        &momentum_learned,
        &mlp_w1,
        &mlp_b1,
        &mlp_w2,
        &mlp_b2,
    );
    for j in 0..golden_delta_w.len() {
        assert!((cpu_dw[j] - golden_delta_w[j]).abs() < 1e-5);
        assert!((auto_dw[j] - golden_delta_w[j]).abs() < 1e-5);
        assert!((cpu_p[j] - golden_param_learned[j]).abs() < 1e-5);
        assert!((auto_p[j] - golden_param_learned[j]).abs() < 1e-5);
        assert!((cpu_m[j] - golden_momentum_learned[j]).abs() < 1e-5);
        assert!((auto_m[j] - golden_momentum_learned[j]).abs() < 1e-5);
    }

    // 4. Linear Anchor CPU
    let cpu_anchor_out = t17::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_18_self_optimising_test_time_adaptation_tent_memo_star_and_spin_cpu_and_auto() -> Result<()>
{
    use ai_kernel::c04_training_methods_and_optimisers::t18_self_optimising_test_time_adaptation_tent_memo_star_and_spin as t18;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let tent_logits = tensors
        .get("tent_logits")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let tent_scale = tensors
        .get("tent_scale")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let tent_shift = tensors
        .get("tent_shift")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_tent_entropy = tensors
        .get("golden_tent_entropy")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_scale_grad = tensors
        .get("golden_scale_grad")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_shift_grad = tensors
        .get("golden_shift_grad")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_scale_next = tensors
        .get("golden_scale_next")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_shift_next = tensors
        .get("golden_shift_next")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let memo_aug_logits = tensors
        .get("memo_aug_logits")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_marginal_p = tensors
        .get("golden_marginal_p")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_memo_entropy = tensors
        .get("golden_memo_entropy")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let logp_win = tensors
        .get("logp_win")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let logp_lose = tensors
        .get("logp_lose")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_spin_loss = tensors
        .get("golden_spin_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. TENT Step CPU & Auto
    let (cpu_s_next, cpu_sh_next, cpu_ent, cpu_gs, cpu_gsh) =
        t18::compute_tent_step_cpu(&tent_logits, &tent_scale, &tent_shift, 4, 4, 0.05);
    let (auto_s_next, auto_sh_next, auto_ent, auto_gs, auto_gsh) =
        t18::compute_tent_step_auto(&kdev, &tent_logits, &tent_scale, &tent_shift, 4, 4, 0.05);

    assert!((cpu_ent - golden_tent_entropy).abs() < 1e-5);
    assert!((auto_ent - golden_tent_entropy).abs() < 1e-5);
    for c in 0..4 {
        assert!((cpu_s_next[c] - golden_scale_next[c]).abs() < 1e-5);
        assert!((auto_s_next[c] - golden_scale_next[c]).abs() < 1e-5);
        assert!((cpu_sh_next[c] - golden_shift_next[c]).abs() < 1e-5);
        assert!((auto_sh_next[c] - golden_shift_next[c]).abs() < 1e-5);
        assert!((cpu_gs[c] - golden_scale_grad[c]).abs() < 1e-5);
        assert!((auto_gs[c] - golden_scale_grad[c]).abs() < 1e-5);
        assert!((cpu_gsh[c] - golden_shift_grad[c]).abs() < 1e-5);
        assert!((auto_gsh[c] - golden_shift_grad[c]).abs() < 1e-5);
    }

    // 2. MEMO CPU & Auto
    let (cpu_mp, cpu_me) = t18::compute_memo_marginal_entropy_cpu(&memo_aug_logits, 3, 4, 1e-8);
    let (auto_mp, auto_me) =
        t18::compute_memo_marginal_entropy_auto(&kdev, &memo_aug_logits, 3, 4, 1e-8);
    assert!((cpu_me - golden_memo_entropy).abs() < 1e-5);
    assert!((auto_me - golden_memo_entropy).abs() < 1e-5);
    for c in 0..4 {
        assert!((cpu_mp[c] - golden_marginal_p[c]).abs() < 1e-5);
        assert!((auto_mp[c] - golden_marginal_p[c]).abs() < 1e-5);
    }

    // 3. SPIN CPU & Auto
    let cpu_spin = t18::compute_spin_loss_cpu(&logp_win, &logp_lose, 0.5);
    let auto_spin = t18::compute_spin_loss_auto(&kdev, &logp_win, &logp_lose, 0.5);
    assert!((cpu_spin - golden_spin_loss).abs() < 1e-5);
    assert!((auto_spin - golden_spin_loss).abs() < 1e-5);

    // 4. Linear Anchor CPU
    let cpu_anchor_out = t18::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_19_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t19_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian as t19;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let h_pos = tensors
        .get("h_pos")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let h_neg = tensors
        .get("h_neg")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_ff_loss = tensors
        .get("golden_ff_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_h_norm = tensors
        .get("golden_h_norm")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let pc_x = tensors
        .get("pc_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let pc_r = tensors
        .get("pc_r")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let pc_w = tensors
        .get("pc_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_pc_r_next = tensors
        .get("golden_pc_r_next")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let top_error = tensors
        .get("top_error")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let dfa_feedback = tensors
        .get("dfa_feedback")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let act_pre = tensors
        .get("act_pre")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_dfa_grad = tensors
        .get("golden_dfa_grad")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let oja_x = tensors
        .get("oja_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let oja_w = tensors
        .get("oja_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_oja_w_next = tensors
        .get("golden_oja_w_next")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. Forward-Forward CPU & Auto
    let (cpu_ff_loss, cpu_h_norm) =
        t19::forward_forward_loss_and_norm_cpu(&h_pos, &h_neg, 4, 8, 2.0, 1e-8);
    let (auto_ff_loss, auto_h_norm) =
        t19::forward_forward_loss_and_norm_auto(&kdev, &h_pos, &h_neg, 4, 8, 2.0, 1e-8);
    assert!((cpu_ff_loss - golden_ff_loss).abs() < 1e-5);
    assert!((auto_ff_loss - golden_ff_loss).abs() < 1e-5);
    for j in 0..golden_h_norm.len() {
        assert!((cpu_h_norm[j] - golden_h_norm[j]).abs() < 1e-5);
        assert!((auto_h_norm[j] - golden_h_norm[j]).abs() < 1e-5);
    }

    // 2. Predictive Coding CPU & Auto
    let cpu_pc_r = t19::predictive_coding_step_cpu(&pc_x, &pc_r, &pc_w, 8, 0.1, 0.01);
    let auto_pc_r = t19::predictive_coding_step_auto(&kdev, &pc_x, &pc_r, &pc_w, 8, 0.1, 0.01);
    for j in 0..8 {
        assert!((cpu_pc_r[j] - golden_pc_r_next[j]).abs() < 1e-5);
        assert!((auto_pc_r[j] - golden_pc_r_next[j]).abs() < 1e-5);
    }

    // 3. DFA CPU & Auto
    let cpu_dfa = t19::dfa_project_error_cpu(&top_error, &dfa_feedback, &act_pre, 4, 8, 4);
    let auto_dfa = t19::dfa_project_error_auto(&kdev, &top_error, &dfa_feedback, &act_pre, 4, 8, 4);
    for j in 0..golden_dfa_grad.len() {
        assert!((cpu_dfa[j] - golden_dfa_grad[j]).abs() < 1e-5);
        assert!((auto_dfa[j] - golden_dfa_grad[j]).abs() < 1e-5);
    }

    // 4. Oja Hebbian CPU & Auto
    let cpu_oja = t19::oja_hebbian_step_cpu(&oja_x, &oja_w, 0.05);
    let auto_oja = t19::oja_hebbian_step_auto(&kdev, &oja_x, &oja_w, 0.05);
    for j in 0..8 {
        assert!((cpu_oja[j] - golden_oja_w_next[j]).abs() < 1e-5);
        assert!((auto_oja[j] - golden_oja_w_next[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t19::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_20_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t20_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck as t20;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let theta = tensors
        .get("theta")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let z = tensors.get("z").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let theta_star = tensors
        .get("theta_star")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_loss_pos = tensors
        .get("golden_loss_pos")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_loss_neg = tensors
        .get("golden_loss_neg")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_dir_deriv = tensors
        .get("golden_dir_deriv")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_theta_next = tensors
        .get("golden_theta_next")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let h = tensors.get("h").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let w_synth = tensors
        .get("w_synth")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b_synth = tensors
        .get("b_synth")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let target_grad = tensors
        .get("target_grad")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_pred_grad = tensors
        .get("golden_pred_grad")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_dni_loss = tensors
        .get("golden_dni_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let hsic_h = tensors
        .get("hsic_h")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let hsic_y = tensors
        .get("hsic_y")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_hsic_score = tensors
        .get("golden_hsic_score")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. MeZO CPU & Auto
    let (cpu_lp, cpu_ln, cpu_dd, cpu_th_next) =
        t20::mezo_step_cpu(&theta, &z, &theta_star, 1e-3, 0.05);
    let (auto_lp, auto_ln, auto_dd, auto_th_next) =
        t20::mezo_step_auto(&kdev, &theta, &z, &theta_star, 1e-3, 0.05);
    assert!((cpu_lp - golden_loss_pos).abs() < 1e-5);
    assert!((auto_lp - golden_loss_pos).abs() < 1e-5);
    assert!((cpu_ln - golden_loss_neg).abs() < 1e-5);
    assert!((auto_ln - golden_loss_neg).abs() < 1e-5);
    assert!((cpu_dd - golden_dir_deriv).abs() < 1e-5);
    assert!((auto_dd - golden_dir_deriv).abs() < 1e-5);
    for j in 0..golden_theta_next.len() {
        assert!((cpu_th_next[j] - golden_theta_next[j]).abs() < 1e-5);
        assert!((auto_th_next[j] - golden_theta_next[j]).abs() < 1e-5);
    }

    // 2. DNI CPU & Auto
    let (cpu_pg, cpu_dl) =
        t20::dni_synthetic_gradient_cpu(&h, &w_synth, &b_synth, &target_grad, 4, 8);
    let (auto_pg, auto_dl) =
        t20::dni_synthetic_gradient_auto(&kdev, &h, &w_synth, &b_synth, &target_grad, 4, 8);
    assert!((cpu_dl - golden_dni_loss).abs() < 1e-5);
    assert!((auto_dl - golden_dni_loss).abs() < 1e-5);
    for j in 0..golden_pred_grad.len() {
        assert!((cpu_pg[j] - golden_pred_grad[j]).abs() < 1e-5);
        assert!((auto_pg[j] - golden_pred_grad[j]).abs() < 1e-5);
    }

    // 3. HSIC CPU & Auto
    let cpu_hsic = t20::hsic_bottleneck_cpu(&hsic_h, &hsic_y, 4, 8, 4);
    let auto_hsic = t20::hsic_bottleneck_auto(&kdev, &hsic_h, &hsic_y, 4, 8, 4);
    assert!((cpu_hsic - golden_hsic_score).abs() < 1e-5);
    assert!((auto_hsic - golden_hsic_score).abs() < 1e-5);

    // 4. Linear Anchor CPU
    let cpu_anchor_out = t20::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_21_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse_cpu_and_auto() -> Result<()>
{
    use ai_kernel::c04_training_methods_and_optimisers::t21_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse as t21;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let n_val = tensors
        .get("n_val")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let d_val = tensors
        .get("d_val")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_chinchilla_loss = tensors
        .get("golden_chinchilla_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_n_opt = tensors
        .get("golden_n_opt")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_d_opt = tensors
        .get("golden_d_opt")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let w_doremi_init = tensors
        .get("w_doremi_init")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let losses_model = tensors
        .get("losses_model")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let losses_ref = tensors
        .get("losses_ref")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_excess_loss = tensors
        .get("golden_excess_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_w_doremi_next = tensors
        .get("golden_w_doremi_next")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let signatures = tensors
        .get("signatures")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_jaccard_matrix = tensors
        .get("golden_jaccard_matrix")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let var_synth = tensors
        .get("var_synth")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let var_real = tensors
        .get("var_real")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_cured_var = tensors
        .get("golden_cured_var")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. Chinchilla CPU & Auto
    let (cpu_l, cpu_n, cpu_d) =
        t21::compute_chinchilla_loss_and_opt_cpu(n_val, d_val, 1.69, 40.0, 50.0, 0.34, 0.28, 0.6);
    let (auto_l, auto_n, auto_d) = t21::compute_chinchilla_loss_and_opt_auto(
        &kdev, n_val, d_val, 1.69, 40.0, 50.0, 0.34, 0.28, 0.6,
    );
    assert!((cpu_l - golden_chinchilla_loss).abs() < 1e-5);
    assert!((auto_l - golden_chinchilla_loss).abs() < 1e-5);
    assert!((cpu_n - golden_n_opt).abs() < 1e-5);
    assert!((auto_n - golden_n_opt).abs() < 1e-5);
    assert!((cpu_d - golden_d_opt).abs() < 1e-5);
    assert!((auto_d - golden_d_opt).abs() < 1e-5);

    // 2. DoReMi CPU & Auto
    let (cpu_ex, cpu_wn) =
        t21::doremi_update_step_cpu(&w_doremi_init, &losses_model, &losses_ref, 1.0);
    let (auto_ex, auto_wn) =
        t21::doremi_update_step_auto(&kdev, &w_doremi_init, &losses_model, &losses_ref, 1.0);
    for j in 0..4 {
        assert!((cpu_ex[j] - golden_excess_loss[j]).abs() < 1e-5);
        assert!((auto_ex[j] - golden_excess_loss[j]).abs() < 1e-5);
        assert!((cpu_wn[j] - golden_w_doremi_next[j]).abs() < 1e-5);
        assert!((auto_wn[j] - golden_w_doremi_next[j]).abs() < 1e-5);
    }

    // 3. MinHash CPU & Auto
    let cpu_jaccard = t21::minhash_pairwise_jaccard_cpu(&signatures, 4, 8);
    let auto_jaccard = t21::minhash_pairwise_jaccard_auto(&kdev, &signatures, 4, 8);
    for j in 0..golden_jaccard_matrix.len() {
        assert!((cpu_jaccard[j] - golden_jaccard_matrix[j]).abs() < 1e-5);
        assert!((auto_jaccard[j] - golden_jaccard_matrix[j]).abs() < 1e-5);
    }

    // 4. Model Collapse CPU & Auto
    let cpu_cured = t21::model_collapse_variance_cure_cpu(var_synth, var_real, 0.25);
    let auto_cured = t21::model_collapse_variance_cure_auto(&kdev, var_synth, var_real, 0.25);
    assert!((cpu_cured - golden_cured_var).abs() < 1e-5);
    assert!((auto_cured - golden_cured_var).abs() < 1e-5);

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t21::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_22_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr_cpu_and_auto()
-> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t22_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr as t22;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let y_grid = tensors
        .get("y_grid")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_total_var = tensors
        .get("golden_total_var")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_v1 = tensors
        .get("golden_v1")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_v2 = tensors
        .get("golden_v2")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_s1 = tensors
        .get("golden_s1")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_s2 = tensors
        .get("golden_s2")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_st1 = tensors
        .get("golden_st1")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_st2 = tensors
        .get("golden_st2")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let w = tensors.get("w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let eta_nominal = tensors
        .get("eta_nominal")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let weight_decay = tensors
        .get("weight_decay")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let grad_norm = tensors
        .get("grad_norm")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_effective_lr = tensors
        .get("golden_effective_lr")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_eq_norm = tensors
        .get("golden_eq_norm")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let eig_max = tensors
        .get("eig_max")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let eta_meta = tensors
        .get("eta_meta")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let steps_t = tensors
        .get("steps_t")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_spectral_multiplier = tensors
        .get("golden_spectral_multiplier")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_accumulated_explosion = tensors
        .get("golden_accumulated_explosion")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. fANOVA & Sobol' CPU & Auto
    let (cpu_tot, cpu_v1, cpu_v2, cpu_s1, cpu_s2, cpu_st1, cpu_st2) =
        t22::compute_fanova_sobol_indices_cpu(&y_grid, 4, 4);
    let (auto_tot, auto_v1, auto_v2, auto_s1, auto_s2, auto_st1, auto_st2) =
        t22::compute_fanova_sobol_indices_auto(&kdev, &y_grid, 4, 4);

    assert!((cpu_tot - golden_total_var).abs() < 1e-5);
    assert!((auto_tot - golden_total_var).abs() < 1e-5);
    assert!((cpu_v1 - golden_v1).abs() < 1e-5);
    assert!((auto_v1 - golden_v1).abs() < 1e-5);
    assert!((cpu_v2 - golden_v2).abs() < 1e-5);
    assert!((auto_v2 - golden_v2).abs() < 1e-5);
    assert!((cpu_s1 - golden_s1).abs() < 1e-5);
    assert!((auto_s1 - golden_s1).abs() < 1e-5);
    assert!((cpu_s2 - golden_s2).abs() < 1e-5);
    assert!((auto_s2 - golden_s2).abs() < 1e-5);
    assert!((cpu_st1 - golden_st1).abs() < 1e-5);
    assert!((auto_st1 - golden_st1).abs() < 1e-5);
    assert!((cpu_st2 - golden_st2).abs() < 1e-5);
    assert!((auto_st2 - golden_st2).abs() < 1e-5);

    // 2. Effective LR & Equilibrium Norm CPU & Auto
    let (cpu_eff_lr, cpu_eq_norm) =
        t22::compute_effective_lr_and_eq_norm_cpu(&w, eta_nominal, weight_decay, grad_norm);
    let (auto_eff_lr, auto_eq_norm) =
        t22::compute_effective_lr_and_eq_norm_auto(&kdev, &w, eta_nominal, weight_decay, grad_norm);
    assert!((cpu_eff_lr - golden_effective_lr).abs() < 1e-5);
    assert!((auto_eff_lr - golden_effective_lr).abs() < 1e-5);
    assert!((cpu_eq_norm - golden_eq_norm).abs() < 1e-5);
    assert!((auto_eq_norm - golden_eq_norm).abs() < 1e-5);

    // 3. Chaotic Meta-Gradient Explosion CPU & Auto
    let (cpu_mult, cpu_accum) =
        t22::compute_chaotic_meta_gradient_explosion_cpu(eig_max, eta_meta, steps_t);
    let (auto_mult, auto_accum) =
        t22::compute_chaotic_meta_gradient_explosion_auto(&kdev, eig_max, eta_meta, steps_t);
    assert!((cpu_mult - golden_spectral_multiplier).abs() < 1e-5);
    assert!((auto_mult - golden_spectral_multiplier).abs() < 1e-5);
    assert!((cpu_accum - golden_accumulated_explosion).abs() < 1e-5);
    assert!((auto_accum - golden_accumulated_explosion).abs() < 1e-5);

    // 4. Linear Anchor Forward CPU
    let cpu_anchor_out = t22::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_23_curriculum_self_paced_learning_and_dataset_distillation_mtt_cpu_and_auto() -> Result<()>
{
    use ai_kernel::c04_training_methods_and_optimisers::t23_curriculum_self_paced_learning_and_dataset_distillation_mtt as t23;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/09_curriculum_self_paced_learning_and_dataset_distillation_mtt.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let losses = tensors
        .get("losses")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let pace_lambda = tensors
        .get("pace_lambda")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_spl_hard_weights = tensors
        .get("golden_spl_hard_weights")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_spl_linear_weights = tensors
        .get("golden_spl_linear_weights")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_weighted_loss_hard = tensors
        .get("golden_weighted_loss_hard")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_weighted_loss_linear = tensors
        .get("golden_weighted_loss_linear")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let difficulties = tensors
        .get("difficulties")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let step_t = tensors
        .get("step_t")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let t_warmup = tensors
        .get("t_warmup")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let c0 = tensors.get("c0").unwrap().flatten_all()?.to_vec1::<f32>()?[0];
    let golden_competence = tensors
        .get("golden_competence")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let golden_active_curriculum_mask = tensors
        .get("golden_active_curriculum_mask")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let grad_real = tensors
        .get("grad_real")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let grad_synth = tensors
        .get("grad_synth")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_grad_matching_loss = tensors
        .get("golden_grad_matching_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let theta_start = tensors
        .get("theta_start")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let theta_student = tensors
        .get("theta_student")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let theta_expert = tensors
        .get("theta_expert")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let golden_mtt_loss = tensors
        .get("golden_mtt_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let anchor_x = tensors
        .get("anchor_x")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_w = tensors
        .get("anchor_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let anchor_b = tensors
        .get("anchor_b")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gold_anchor_out = tensors
        .get("golden_anchor_out")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. SPL CPU & Auto
    let (cpu_hw, cpu_lw, cpu_lh, cpu_ll) =
        t23::compute_spl_weights_and_loss_cpu(&losses, pace_lambda);
    let (auto_hw, auto_lw, auto_lh, auto_ll) =
        t23::compute_spl_weights_and_loss_auto(&kdev, &losses, pace_lambda);

    for j in 0..golden_spl_hard_weights.len() {
        assert!((cpu_hw[j] - golden_spl_hard_weights[j]).abs() < 1e-5);
        assert!((auto_hw[j] - golden_spl_hard_weights[j]).abs() < 1e-5);
        assert!((cpu_lw[j] - golden_spl_linear_weights[j]).abs() < 1e-5);
        assert!((auto_lw[j] - golden_spl_linear_weights[j]).abs() < 1e-5);
    }
    assert!((cpu_lh - golden_weighted_loss_hard).abs() < 1e-5);
    assert!((auto_lh - golden_weighted_loss_hard).abs() < 1e-5);
    assert!((cpu_ll - golden_weighted_loss_linear).abs() < 1e-5);
    assert!((auto_ll - golden_weighted_loss_linear).abs() < 1e-5);

    // 2. Curriculum Competence CPU & Auto
    let (cpu_comp, cpu_mask) =
        t23::compute_competence_and_curriculum_mask_cpu(&difficulties, step_t, t_warmup, c0);
    let (auto_comp, auto_mask) = t23::compute_competence_and_curriculum_mask_auto(
        &kdev,
        &difficulties,
        step_t,
        t_warmup,
        c0,
    );
    assert!((cpu_comp - golden_competence).abs() < 1e-5);
    assert!((auto_comp - golden_competence).abs() < 1e-5);
    for j in 0..golden_active_curriculum_mask.len() {
        assert!((cpu_mask[j] - golden_active_curriculum_mask[j]).abs() < 1e-5);
        assert!((auto_mask[j] - golden_active_curriculum_mask[j]).abs() < 1e-5);
    }

    // 3. Gradient Matching CPU & Auto
    let cpu_gm = t23::compute_gradient_matching_loss_cpu(&grad_synth, &grad_real);
    let auto_gm = t23::compute_gradient_matching_loss_auto(&kdev, &grad_synth, &grad_real);
    assert!((cpu_gm - golden_grad_matching_loss).abs() < 1e-5);
    assert!((auto_gm - golden_grad_matching_loss).abs() < 1e-5);

    // 4. Trajectory Matching (MTT) CPU & Auto
    let cpu_mtt =
        t23::compute_trajectory_matching_loss_cpu(&theta_start, &theta_student, &theta_expert);
    let auto_mtt = t23::compute_trajectory_matching_loss_auto(
        &kdev,
        &theta_start,
        &theta_student,
        &theta_expert,
    );
    assert!((cpu_mtt - golden_mtt_loss).abs() < 1e-5);
    assert!((auto_mtt - golden_mtt_loss).abs() < 1e-5);

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t23::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}
