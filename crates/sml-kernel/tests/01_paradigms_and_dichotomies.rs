use candle_core::{Device, safetensors::load};
use sml_kernel::{
    KernelDevice,
    c01_paradigms_and_dichotomies::{
        t01_generative_vs_discriminative_classifiers as t01,
        t02_tree_vs_kernel_vs_neural_inductive_biases as t02,
        t03_discrete_vs_continuous_latents_and_gradient_estimators as t03,
        t04_deterministic_vs_stochastic_and_mixture_density_networks as t04,
        t05_parametric_vs_nonparametric_and_neural_processes as t05,
        t06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic as t06,
        t07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds as t07,
        t08_computation_spaces_latent_reasoning_and_jepa as t08,
        t09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd as t09,
        t10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games as t10,
        t11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes as t11,
        t12_inductive_vs_transductive_and_online_regret_vs_batch_iid as t12,
    },
};

mod common;

#[test]
fn test_01_generative_vs_discriminative_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t01::discriminative_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t01::discriminative_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // GDA logits test against layers.gda_logits
    if let Some(gda_logits) = tensors.get("layers.gda_logits") {
        let gda_gold: Vec<f32> = gda_logits.to_vec2::<f32>()?.into_iter().flatten().collect();
        let w_gda: Vec<f32> = tensors
            .get("layers.gda_weight")
            .unwrap()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let b_gda = tensors.get("layers.gda_bias").unwrap().to_vec1::<f32>()?;

        let calc_gda = t01::predict_logits_gda(
            &KernelDevice::Cpu,
            &x_flat,
            &w_gda,
            &b_gda,
            batch,
            in_dim,
            out_dim,
        );
        for (i, (&calc, &gold)) in calc_gda.iter().zip(gda_gold.iter()).enumerate() {
            assert!(
                (calc - gold).abs() < 1e-5,
                "GDA error at {i}: {calc} vs {gold}"
            );
        }
    }

    Ok(())
}

#[test]
fn test_02_tree_vs_kernel_vs_neural_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t02::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t02::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // Verify soft decision tree
    let w_route: Vec<f32> = tensors
        .get("layers.tree_routing_weights")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let thresh = tensors
        .get("layers.tree_thresholds")
        .unwrap()
        .to_vec1::<f32>()?;
    let temp = tensors
        .get("layers.tree_temperature")
        .unwrap()
        .to_vec1::<f32>()?[0];
    let w_leaf: Vec<f32> = tensors
        .get("layers.tree_leaf_weights")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let tree_gold: Vec<f32> = tensors
        .get("layers.tree_output")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let tree_calc = t02::soft_decision_tree_cpu(
        &x_flat, &w_route, &thresh, temp, &w_leaf, batch, in_dim, out_dim,
    );
    for (i, (&calc, &gold)) in tree_calc.iter().zip(tree_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Tree error at {i}: {calc} vs {gold}"
        );
    }

    // Verify RBF kernel ridge
    let support: Vec<f32> = tensors
        .get("layers.kernel_support")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let dual_coeffs: Vec<f32> = tensors
        .get("layers.kernel_dual_coeffs")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let gamma = tensors
        .get("layers.kernel_gamma")
        .unwrap()
        .to_vec1::<f32>()?[0];
    let k_bias = tensors
        .get("layers.kernel_bias")
        .unwrap()
        .to_vec1::<f32>()?;
    let k_gold: Vec<f32> = tensors
        .get("layers.kernel_output")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let k_calc = t02::rbf_kernel_ridge_cpu(
        &x_flat,
        &support,
        &dual_coeffs,
        gamma,
        &k_bias,
        batch,
        support.len() / in_dim,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in k_calc.iter().zip(k_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Kernel ridge error at {i}: {calc} vs {gold}"
        );
    }

    // Verify MLP classifier
    let mlp_w1: Vec<f32> = tensors
        .get("layers.mlp_w1")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let mlp_b1 = tensors.get("layers.mlp_b1").unwrap().to_vec1::<f32>()?;
    let mlp_w2: Vec<f32> = tensors
        .get("layers.mlp_w2")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let mlp_b2 = tensors.get("layers.mlp_b2").unwrap().to_vec1::<f32>()?;
    let mlp_gold: Vec<f32> = tensors
        .get("layers.mlp_output")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let mlp_calc = t02::mlp_classifier_cpu(
        &x_flat,
        &mlp_w1,
        &mlp_b1,
        &mlp_w2,
        &mlp_b2,
        batch,
        in_dim,
        mlp_b1.len(),
        out_dim,
    );
    for (i, (&calc, &gold)) in mlp_calc.iter().zip(mlp_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "MLP error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_03_discrete_vs_continuous_latents_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t03::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t03::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // Verify Gaussian Reparameterization and KL
    let w_mu: Vec<f32> = tensors
        .get("layers.w_mu")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_mu = tensors.get("layers.b_mu").unwrap().to_vec1::<f32>()?;
    let w_logvar: Vec<f32> = tensors
        .get("layers.w_logvar")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_logvar = tensors.get("layers.b_logvar").unwrap().to_vec1::<f32>()?;
    let eps: Vec<f32> = tensors
        .get("layers.eps")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let z_gold: Vec<f32> = tensors
        .get("layers.continuous_z")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let kl_gold = tensors
        .get("layers.kl_divergence")
        .unwrap()
        .to_vec1::<f32>()?;

    let latent_dim = b_mu.len();
    let (mu, logvar) = t03::gaussian_encode_cpu(
        &x_flat, &w_mu, &b_mu, &w_logvar, &b_logvar, batch, in_dim, latent_dim,
    );
    let z_calc = t03::gaussian_reparameterize_cpu(&mu, &logvar, &eps, batch, latent_dim);
    for (i, (&calc, &gold)) in z_calc.iter().zip(z_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "z error at {i}: {calc} vs {gold}"
        );
    }

    let kl_calc = t03::gaussian_kl_divergence_cpu(&mu, &logvar, batch, latent_dim);
    for (i, (&calc, &gold)) in kl_calc.iter().zip(kl_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "KL error at {i}: {calc} vs {gold}"
        );
    }

    // Verify Gumbel Softmax
    let w_logits: Vec<f32> = tensors
        .get("layers.w_logits")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_logits = tensors.get("layers.b_logits").unwrap().to_vec1::<f32>()?;
    let gumbel_noise: Vec<f32> = tensors
        .get("layers.gumbel_noise")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let temp = tensors
        .get("layers.temperature")
        .unwrap()
        .to_vec1::<f32>()?[0];
    let gumbel_gold: Vec<f32> = tensors
        .get("layers.gumbel_soft")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let gumbel_calc = t03::gumbel_softmax_sample_cpu(
        &x_flat,
        &w_logits,
        &b_logits,
        &gumbel_noise,
        temp,
        batch,
        in_dim,
        b_logits.len(),
    );
    for (i, (&calc, &gold)) in gumbel_calc.iter().zip(gumbel_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Gumbel error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_04_deterministic_vs_stochastic_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t04::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t04::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // Verify deterministic forward
    let w_h: Vec<f32> = tensors
        .get("layers.w_h")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_h = tensors.get("layers.b_h").unwrap().to_vec1::<f32>()?;
    let w_det: Vec<f32> = tensors
        .get("layers.w_det")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_det = tensors.get("layers.b_det").unwrap().to_vec1::<f32>()?;
    let y_det_gold: Vec<f32> = tensors
        .get("layers.y_det")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let y_det_calc = t04::deterministic_forward_cpu(
        &x_flat,
        &w_h,
        &b_h,
        &w_det,
        &b_det,
        batch,
        in_dim,
        b_h.len(),
        b_det.len(),
    );
    for (i, (&calc, &gold)) in y_det_calc.iter().zip(y_det_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Det forward error at {i}: {calc} vs {gold}"
        );
    }

    // Verify MDN log probs
    let w_pi: Vec<f32> = tensors
        .get("layers.w_pi")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_pi = tensors.get("layers.b_pi").unwrap().to_vec1::<f32>()?;
    let w_mu: Vec<f32> = tensors
        .get("layers.w_mu")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_mu = tensors.get("layers.b_mu").unwrap().to_vec1::<f32>()?;
    let w_sigma: Vec<f32> = tensors
        .get("layers.w_sigma")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_sigma = tensors.get("layers.b_sigma").unwrap().to_vec1::<f32>()?;
    let sample_y = tensors
        .get("layers.sample_y")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let log_probs_gold = tensors
        .get("layers.mdn_log_probs")
        .unwrap()
        .to_vec1::<f32>()?;

    let (pi, mu, sigma) = t04::mdn_forward_cpu(
        &x_flat,
        &w_h,
        &b_h,
        &w_pi,
        &b_pi,
        &w_mu,
        &b_mu,
        &w_sigma,
        &b_sigma,
        batch,
        in_dim,
        b_h.len(),
        b_pi.len(),
    );
    let log_probs_calc = t04::mdn_log_prob_cpu(&pi, &mu, &sigma, &sample_y, batch, b_pi.len());
    for (i, (&calc, &gold)) in log_probs_calc.iter().zip(log_probs_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-4,
            "MDN log prob error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_05_parametric_vs_nonparametric_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t05::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t05::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // Verify parametric MLP
    let w_mlp1: Vec<f32> = tensors
        .get("layers.w_mlp1")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_mlp1 = tensors.get("layers.b_mlp1").unwrap().to_vec1::<f32>()?;
    let w_mlp2: Vec<f32> = tensors
        .get("layers.w_mlp2")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_mlp2 = tensors.get("layers.b_mlp2").unwrap().to_vec1::<f32>()?;
    let x_target = tensors.get("layers.x_target").unwrap().to_vec3::<f32>()?;
    let y_param_gold: Vec<f32> = tensors
        .get("layers.y_parametric")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();

    let x_target_flat: Vec<f32> = x_target.into_iter().flatten().flatten().collect();
    let y_param_calc = t05::parametric_forward_cpu(
        &x_target_flat,
        &w_mlp1,
        &b_mlp1,
        &w_mlp2,
        &b_mlp2,
        batch * 4,
        2,
        b_mlp1.len(),
        1,
    );
    for (i, (&calc, &gold)) in y_param_calc.iter().zip(y_param_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Parametric error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_06_ai_by_purpose_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t06::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t06::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // Verify Cost predictor, regret and SPO+ loss
    let w_cost1: Vec<f32> = tensors
        .get("layers.w_cost1")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_cost1 = tensors.get("layers.b_cost1").unwrap().to_vec1::<f32>()?;
    let w_cost2: Vec<f32> = tensors
        .get("layers.w_cost2")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_cost2 = tensors.get("layers.b_cost2").unwrap().to_vec1::<f32>()?;
    let c_pred_gold: Vec<f32> = tensors
        .get("layers.c_pred")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let c_star: Vec<f32> = tensors
        .get("layers.c_star")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let regret_gold = tensors
        .get("layers.decision_regret")
        .unwrap()
        .to_vec1::<f32>()?;
    let spo_gold = tensors
        .get("layers.spo_plus_loss")
        .unwrap()
        .to_vec1::<f32>()?;

    let c_pred_calc = t06::cost_predictor_forward_cpu(
        &x_flat,
        &w_cost1,
        &b_cost1,
        &w_cost2,
        &b_cost2,
        batch,
        in_dim,
        b_cost1.len(),
        b_cost2.len(),
    );
    for (i, (&calc, &gold)) in c_pred_calc.iter().zip(c_pred_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "c_pred error at {i}: {calc} vs {gold}"
        );
    }

    let regret_calc = t06::decision_regret_cpu(&c_pred_calc, &c_star, batch, b_cost2.len());
    for (i, (&calc, &gold)) in regret_calc.iter().zip(regret_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "regret error at {i}: {calc} vs {gold}"
        );
    }

    let spo_calc = t06::spo_plus_loss_cpu(&c_pred_calc, &c_star, batch, b_cost2.len());
    for (i, (&calc, &gold)) in spo_calc.iter().zip(spo_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "SPO+ error at {i}: {calc} vs {gold}"
        );
    }

    // Verify Causal
    let t_vec: Vec<f32> = tensors
        .get("layers.treatment_t")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let w_tau: Vec<f32> = tensors
        .get("layers.w_tau")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_tau = tensors.get("layers.b_tau").unwrap().to_vec1::<f32>()?;
    let w_mu0: Vec<f32> = tensors
        .get("layers.w_mu0")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_mu0 = tensors.get("layers.b_mu0").unwrap().to_vec1::<f32>()?;
    let y_fac_gold: Vec<f32> = tensors
        .get("layers.y_factual")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let y_fac_calc = t06::causal_forward_cpu(
        &x_flat, &t_vec, &w_tau, &b_tau, &w_mu0, &b_mu0, batch, in_dim,
    );
    for (i, (&calc, &gold)) in y_fac_calc.iter().zip(y_fac_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Causal error at {i}: {calc} vs {gold}"
        );
    }

    // Verify Symbolic EQL
    let w_eql: Vec<f32> = tensors
        .get("layers.w_eql")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_eql = tensors.get("layers.b_eql").unwrap().to_vec1::<f32>()?;
    let y_sym_gold: Vec<f32> = tensors
        .get("layers.y_symbolic")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let y_sym_calc = t06::symbolic_forward_cpu(&x_flat, &w_eql, &b_eql, batch);
    for (i, (&calc, &gold)) in y_sym_calc.iter().zip(y_sym_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Symbolic error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_07_input_output_spaces_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t07::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t07::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // Verify 6D rotation matrix
    let rot_gold: Vec<f32> = tensors
        .get("layers.rot_matrices")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let rot_calc = t07::rotation_6d_to_matrix_cpu(&x_flat, batch);
    for (i, (&calc, &gold)) in rot_calc.iter().zip(rot_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "6D rotation error at {i}: {calc} vs {gold}"
        );
    }

    // Verify Fourier coordinate encoding
    let coords: Vec<f32> = tensors
        .get("layers.coords_2d")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let fourier_gold: Vec<f32> = tensors
        .get("layers.coords_fourier")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let fourier_calc = t07::fourier_coordinate_encode_cpu(&coords, batch, 2, 4);
    for (i, (&calc, &gold)) in fourier_calc.iter().zip(fourier_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Fourier coords error at {i}: {calc} vs {gold}"
        );
    }

    // Verify ViT patch embeddings
    let patches: Vec<f32> = tensors
        .get("layers.patches")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let w_patch: Vec<f32> = tensors
        .get("layers.w_patch")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_patch = tensors.get("layers.b_patch").unwrap().to_vec1::<f32>()?;
    let patch_embed_gold: Vec<f32> = tensors
        .get("layers.patch_embeddings")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();

    let patch_embed_calc =
        t07::patch_embedding_forward_cpu(&patches, &w_patch, &b_patch, batch, 4, 4, b_patch.len());
    for (i, (&calc, &gold)) in patch_embed_calc
        .iter()
        .zip(patch_embed_gold.iter())
        .enumerate()
    {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Patch embedding error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_08_computation_spaces_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/08_computation_spaces_latent_reasoning_and_jepa.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    // 1. Canonical Linear Forward
    let y_cpu = t08::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t08::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Continuous Latent Reasoning
    let w_proj: Vec<f32> = tensors
        .get("layers.w_proj")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_proj = tensors.get("layers.b_proj").unwrap().to_vec1::<f32>()?;
    let w_z: Vec<f32> = tensors
        .get("layers.w_z")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let w_anc: Vec<f32> = tensors
        .get("layers.w_anc")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_z = tensors.get("layers.b_z").unwrap().to_vec1::<f32>()?;
    let traj_gold: Vec<f32> = tensors
        .get("layers.latent_trajectories")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let final_gold: Vec<f32> = tensors
        .get("layers.final_thought")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let dim = b_z.len();
    let steps = 3;
    let (traj_calc, final_calc) = t08::continuous_latent_reasoning_cpu(
        &x_flat, &w_proj, &b_proj, &w_z, &w_anc, &b_z, batch, in_dim, dim, steps,
    );

    for (i, (&calc, &gold)) in traj_calc.iter().zip(traj_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Traj error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in final_calc.iter().zip(final_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Final thought error at {i}: {calc} vs {gold}"
        );
    }

    // 3. PonderNet Halting Hazards
    let w_h = tensors
        .get("layers.w_h")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let b_h = tensors.get("layers.b_h").unwrap().to_vec1::<f32>()?;
    let halting_gold: Vec<f32> = tensors
        .get("layers.halting_probs")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let halting_calc = t08::ponder_halting_cpu(&traj_calc, &w_h, &b_h, batch, steps, dim);
    for (i, (&calc, &gold)) in halting_calc.iter().zip(halting_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Halting error at {i}: {calc} vs {gold}"
        );
    }

    // 4. JEPA Latent-Space Predictor & Loss
    let w_p1: Vec<f32> = tensors
        .get("layers.w_p1")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_p1 = tensors.get("layers.b_p1").unwrap().to_vec1::<f32>()?;
    let w_p2: Vec<f32> = tensors
        .get("layers.w_p2")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let b_p2 = tensors.get("layers.b_p2").unwrap().to_vec1::<f32>()?;
    let jepa_pred_gold: Vec<f32> = tensors
        .get("layers.jepa_pred")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let target_sy: Vec<f32> = tensors
        .get("layers.target_sy")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let jepa_loss_gold = tensors.get("layers.jepa_error").unwrap().to_vec1::<f32>()?;

    let jepa_pred_calc = t08::jepa_predict_cpu(&final_calc, &w_p1, &b_p1, &w_p2, &b_p2, batch, dim);
    for (i, (&calc, &gold)) in jepa_pred_calc.iter().zip(jepa_pred_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "JEPA pred error at {i}: {calc} vs {gold}"
        );
    }

    let jepa_loss_calc = t08::jepa_loss_cpu(&jepa_pred_calc, &target_sy, batch, dim);
    for (i, (&calc, &gold)) in jepa_loss_calc.iter().zip(jepa_loss_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "JEPA loss error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_09_non_euclidean_geometries_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd.safetensors",
    );
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let w = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    // 1. Canonical Linear Forward
    let y_cpu = t09::linear_forward(
        &KernelDevice::Cpu,
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU error at {i}: {calc} vs {gold}"
        );
    }

    let y_auto = t09::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Hyperbolic Poincaré Ball
    let c_val = tensors.get("layers.c").unwrap().to_vec1::<f32>()?[0];
    let v_flat: Vec<f32> = tensors
        .get("layers.euclidean_v")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let exp_gold: Vec<f32> = tensors
        .get("layers.exp_map_0")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let exp_calc = t09::poincare_exp_zero_cpu(&v_flat, batch, 2, c_val);
    for (i, (&calc, &gold)) in exp_calc.iter().zip(exp_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Exp map error at {i}: {calc} vs {gold}"
        );
    }

    let x_hyp_flat: Vec<f32> = tensors
        .get("layers.poincare_x")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let log_gold: Vec<f32> = tensors
        .get("layers.log_map_0")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let log_calc = t09::poincare_log_zero_cpu(&x_hyp_flat, batch, 2, c_val);
    for (i, (&calc, &gold)) in log_calc.iter().zip(log_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Log map error at {i}: {calc} vs {gold}"
        );
    }

    let y_hyp_flat: Vec<f32> = tensors
        .get("layers.poincare_y")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let mobius_gold: Vec<f32> = tensors
        .get("layers.mobius_add")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let mobius_calc = t09::poincare_mobius_add_cpu(&x_hyp_flat, &y_hyp_flat, batch, 2, c_val);
    for (i, (&calc, &gold)) in mobius_calc.iter().zip(mobius_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Mobius add error at {i}: {calc} vs {gold}"
        );
    }

    let dist_gold = tensors
        .get("layers.poincare_dist")
        .unwrap()
        .to_vec1::<f32>()?;
    let dist_calc = t09::poincare_distance_cpu(&x_hyp_flat, &y_hyp_flat, batch, 2, c_val);
    for (i, (&calc, &gold)) in dist_calc.iter().zip(dist_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Poincare dist error at {i}: {calc} vs {gold}"
        );
    }

    // 3. Spherical Geometry
    let p_flat: Vec<f32> = tensors
        .get("layers.sphere_p")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let q_flat: Vec<f32> = tensors
        .get("layers.sphere_q")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let s_dist_gold = tensors
        .get("layers.sphere_dist")
        .unwrap()
        .to_vec1::<f32>()?;
    let s_dist_calc = t09::spherical_distance_cpu(&p_flat, &q_flat, batch, 3);
    for (i, (&calc, &gold)) in s_dist_calc.iter().zip(s_dist_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Sphere dist error at {i}: {calc} vs {gold}"
        );
    }

    let align_gold = tensors
        .get("layers.alignment_loss")
        .unwrap()
        .to_vec1::<f32>()?;
    let align_calc = t09::spherical_alignment_cpu(&p_flat, &q_flat, batch, 3);
    for (i, (&calc, &gold)) in align_calc.iter().zip(align_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Sphere align error at {i}: {calc} vs {gold}"
        );
    }

    // 4. SPD Log-Euclidean distance
    let spd_p_flat: Vec<f32> = tensors
        .get("layers.spd_P")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let spd_q_flat: Vec<f32> = tensors
        .get("layers.spd_Q")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let spd_dist_gold = tensors
        .get("layers.log_euclidean_dist")
        .unwrap()
        .to_vec1::<f32>()?;
    let spd_dist_calc = t09::spd_log_euclidean_distance_2x2_cpu(&spd_p_flat, &spd_q_flat, batch);
    for (i, (&calc, &gold)) in spd_dist_calc.iter().zip(spd_dist_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "SPD dist error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_10_decision_hierarchy_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.safetensors",
    );
    let tensors = load(path, &device)?;

    // 1. Level 0 Bayes Risk on CPU
    let probs_flat: Vec<f32> = tensors
        .get("layers.l0_probs")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let cost_flat: Vec<f32> = tensors
        .get("layers.l0_cost_matrix")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let risk_gold: Vec<f32> = tensors
        .get("layers.l0_expected_risk")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let acts_gold = tensors
        .get("layers.l0_best_actions")
        .unwrap()
        .to_vec1::<f32>()?;
    let rej_gold = tensors
        .get("layers.l0_reject_mask")
        .unwrap()
        .to_vec1::<f32>()?;

    let (risk_calc, acts_calc, rej_calc) =
        t10::bayes_decision_rule_cpu(&probs_flat, &cost_flat, 4, 3, 3, 0.5);
    for (i, (&calc, &gold)) in risk_calc.iter().zip(risk_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Risk error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in acts_calc.iter().zip(acts_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Acts error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in rej_calc.iter().zip(rej_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Rej error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Level 1 Bandits OPE on CPU
    let tp = tensors
        .get("layers.l1_target_policy_probs")
        .unwrap()
        .to_vec1::<f32>()?;
    let lp = tensors
        .get("layers.l1_logged_propensities")
        .unwrap()
        .to_vec1::<f32>()?;
    let rew = tensors.get("layers.l1_rewards").unwrap().to_vec1::<f32>()?;
    let q = tensors
        .get("layers.l1_baseline_q")
        .unwrap()
        .to_vec1::<f32>()?;
    let w_gold = tensors
        .get("layers.l1_ips_weights")
        .unwrap()
        .to_vec1::<f32>()?;
    let dr_gold = tensors
        .get("layers.l1_doubly_robust_estimates")
        .unwrap()
        .to_vec1::<f32>()?;

    let (w_calc, dr_calc) = t10::bandits_ope_cpu(&tp, &lp, &rew, &q, 4);
    for (i, (&calc, &gold)) in w_calc.iter().zip(w_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Weight error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in dr_calc.iter().zip(dr_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "DR error at {i}: {calc} vs {gold}"
        );
    }

    // 3. Level 2 MDP Bellman on CPU
    let v_init = tensors.get("layers.l2_v_init").unwrap().to_vec1::<f32>()?;
    let trans_flat: Vec<f32> = tensors
        .get("layers.l2_transitions")
        .unwrap()
        .to_vec3::<f32>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let rew_flat: Vec<f32> = tensors
        .get("layers.l2_rewards")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let q_gold: Vec<f32> = tensors
        .get("layers.l2_q_values")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let v_gold = tensors
        .get("layers.l2_v_backup")
        .unwrap()
        .to_vec1::<f32>()?;

    let (q_calc, v_calc) = t10::mdp_bellman_cpu(&v_init, &trans_flat, &rew_flat, 4, 2, 0.9);
    for (i, (&calc, &gold)) in q_calc.iter().zip(q_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Q error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in v_calc.iter().zip(v_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "V error at {i}: {calc} vs {gold}"
        );
    }

    // 4. Level 3 POMDP Belief on CPU
    let prior = tensors
        .get("layers.l3_prior_belief")
        .unwrap()
        .to_vec1::<f32>()?;
    let trans_p_flat: Vec<f32> = tensors
        .get("layers.l3_transition")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let obs = tensors
        .get("layers.l3_observation_likelihood")
        .unwrap()
        .to_vec1::<f32>()?;
    let post_gold = tensors
        .get("layers.l3_posterior_belief")
        .unwrap()
        .to_vec1::<f32>()?;

    let post_calc = t10::pomdp_belief_cpu(&prior, &trans_p_flat, &obs, 3);
    for (i, (&calc, &gold)) in post_calc.iter().zip(post_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "POMDP post error at {i}: {calc} vs {gold}"
        );
    }

    // 5. Level 5 Auction on CPU
    let bids_flat: Vec<f32> = tensors
        .get("layers.l5_bids")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let alloc_gold: Vec<f32> = tensors
        .get("layers.l5_allocations")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let pay_gold: Vec<f32> = tensors
        .get("layers.l5_payments")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let (alloc_calc, pay_calc) = t10::differentiable_auction_cpu(&bids_flat, 4, 0.5);
    for (i, (&calc, &gold)) in alloc_calc.iter().zip(alloc_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auction alloc error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in pay_calc.iter().zip(pay_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auction pay error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_11_statistical_inference_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.safetensors",
    );
    let tensors = load(path, &device)?;

    // 1. Frequentist Fisher & Cramér-Rao on CPU
    let x_data_flat: Vec<f32> = tensors
        .get("x_data")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let theta_mle_gold = tensors.get("theta_mle").unwrap().to_vec1::<f32>()?;
    let fisher_gold = tensors.get("fisher_diag").unwrap().to_vec1::<f32>()?;
    let crlb_gold = tensors
        .get("cramer_rao_lower_bound")
        .unwrap()
        .to_vec1::<f32>()?;
    let lrt_gold = tensors.get("lrt_stat").unwrap().to_scalar::<f32>()?;

    let theta_0 = vec![0.0f32; 4];
    let (mle_calc, fisher_calc, crlb_calc, lrt_calc) =
        t11::frequentist_fisher_cpu(&x_data_flat, 16, 4, 0.5, &theta_0);

    for (i, (&calc, &gold)) in mle_calc.iter().zip(theta_mle_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "MLE error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in fisher_calc.iter().zip(fisher_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Fisher error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in crlb_calc.iter().zip(crlb_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CRLB error at {i}: {calc} vs {gold}"
        );
    }
    assert!(
        (lrt_calc - lrt_gold).abs() < 1e-4,
        "LRT error: {lrt_calc} vs {lrt_gold}"
    );

    // 2. Bayesian Conjugate & Laplace Occam Factor on CPU
    let mu_0 = tensors.get("mu_0").unwrap().to_vec1::<f32>()?;
    let mu_post_gold = tensors.get("mu_post").unwrap().to_vec1::<f32>()?;
    let sig_post_gold = tensors.get("sigma_post_vec").unwrap().to_vec1::<f32>()?;
    let laplace_gold = tensors
        .get("laplace_log_marginal")
        .unwrap()
        .to_scalar::<f32>()?;
    let occam_gold = tensors.get("occam_penalty").unwrap().to_scalar::<f32>()?;

    let (mu_calc, sig_calc, lap_calc, occ_calc) =
        t11::bayesian_conjugate_laplace_cpu(&x_data_flat, &mu_0, 16, 4, 1.0, 0.5);
    for (i, (&calc, &gold)) in mu_calc.iter().zip(mu_post_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Mu post error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in sig_calc.iter().zip(sig_post_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Sig post error at {i}: {calc} vs {gold}"
        );
    }
    assert!(
        (lap_calc - laplace_gold).abs() < 1e-4,
        "Laplace error: {lap_calc} vs {laplace_gold}"
    );
    assert!(
        (occ_calc - occam_gold).abs() < 1e-4,
        "Occam error: {occ_calc} vs {occam_gold}"
    );

    // 3. James-Stein Shrinkage on CPU and Auto
    let y_obs = tensors.get("y_obs").unwrap().to_vec1::<f32>()?;
    let th_js_gold = tensors.get("theta_js").unwrap().to_vec1::<f32>()?;
    let th_js_pos_gold = tensors.get("theta_js_pos").unwrap().to_vec1::<f32>()?;
    let c_shrink_gold = tensors.get("c_shrink").unwrap().to_scalar::<f32>()?;

    let (th_js_cpu, th_js_pos_cpu, c_sh_cpu) = t11::james_stein_cpu(&y_obs, 1.0);
    for (i, (&calc, &gold)) in th_js_cpu.iter().zip(th_js_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "JS CPU error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in th_js_pos_cpu.iter().zip(th_js_pos_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "JS pos CPU error at {i}: {calc} vs {gold}"
        );
    }
    assert!(
        (c_sh_cpu - c_shrink_gold).abs() < 1e-5,
        "C shrink error: {c_sh_cpu} vs {c_shrink_gold}"
    );

    let (th_js_auto, th_js_pos_auto, c_sh_auto) = t11::james_stein_auto(None, None, &y_obs, 1.0);
    for (i, (&calc, &gold)) in th_js_auto.iter().zip(th_js_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "JS Auto error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in th_js_pos_auto.iter().zip(th_js_pos_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "JS pos Auto error at {i}: {calc} vs {gold}"
        );
    }
    assert!(
        (c_sh_auto - c_shrink_gold).abs() < 1e-5,
        "C shrink auto error"
    );

    // 4. PAC-Bayes on CPU
    let mu_q = tensors.get("mu_q").unwrap().to_vec1::<f32>()?;
    let sig_q = tensors.get("sigma_q").unwrap().to_vec1::<f32>()?;
    let emp_risk = tensors.get("emp_risk").unwrap().to_scalar::<f32>()?;
    let kl_gold = tensors.get("kl_div").unwrap().to_scalar::<f32>()?;
    let pac_gold = tensors.get("pac_bound").unwrap().to_scalar::<f32>()?;

    let (kl_calc, pac_calc) = t11::pac_bayes_cpu(&mu_q, &sig_q, emp_risk, 16, 1.0, 0.05);
    assert!(
        (kl_calc - kl_gold).abs() < 1e-5,
        "KL error: {kl_calc} vs {kl_gold}"
    );
    assert!(
        (pac_calc - pac_gold).abs() < 1e-5,
        "PAC error: {pac_calc} vs {pac_gold}"
    );

    // 5. Anytime-Valid Martingale on CPU
    let e_vals = tensors.get("e_values").unwrap().to_vec1::<f32>()?;
    let mart_gold = tensors.get("martingale").unwrap().to_vec1::<f32>()?;
    let thresh_gold = tensors.get("ville_threshold").unwrap().to_scalar::<f32>()?;

    let (mart_calc, thresh_calc) = t11::anytime_valid_martingale_cpu(&e_vals, 0.05);
    for (i, (&calc, &gold)) in mart_calc.iter().zip(mart_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Martingale error at {i}: {calc} vs {gold}"
        );
    }
    assert!(
        (thresh_calc - thresh_gold).abs() < 1e-5,
        "Threshold error: {thresh_calc} vs {thresh_gold}"
    );

    Ok(())
}

#[test]
fn test_12_inductive_vs_transductive_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.safetensors",
    );
    let tensors = load(path, &device)?;

    // 1. Transductive Label Spreading on CPU
    let laplacian_s_flat: Vec<f32> = tensors
        .get("laplacian_s")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let y_init_flat: Vec<f32> = tensors
        .get("y_init")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let f_star_gold: Vec<f32> = tensors
        .get("f_star")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let f_probs_gold: Vec<f32> = tensors
        .get("f_probs")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let t_preds_gold = tensors
        .get("transductive_preds")
        .unwrap()
        .to_vec1::<f32>()?;

    let (f_star_calc, f_probs_calc, t_preds_calc) =
        t12::transductive_label_spreading_cpu(&laplacian_s_flat, &y_init_flat, 8, 2, 0.8);
    for (i, (&calc, &gold)) in f_star_calc.iter().zip(f_star_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "F* error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in f_probs_calc.iter().zip(f_probs_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "F probs error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in t_preds_calc.iter().zip(t_preds_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Transductive preds error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Inductive Predictor on CPU
    let node_coords = tensors.get("node_coords").unwrap().to_vec2::<f32>()?;
    let x_unlabeled_flat: Vec<f32> = node_coords[4..].iter().flatten().copied().collect();
    let w_ind = tensors.get("w_ind").unwrap().to_vec1::<f32>()?;
    let ind_scores_gold = tensors.get("inductive_scores").unwrap().to_vec1::<f32>()?;
    let ind_preds_gold = tensors.get("inductive_preds").unwrap().to_vec1::<f32>()?;

    let (scores_calc, preds_calc) = t12::inductive_predictor_cpu(&x_unlabeled_flat, &w_ind, 4, 2);
    for (i, (&calc, &gold)) in scores_calc.iter().zip(ind_scores_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Inductive scores error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in preds_calc.iter().zip(ind_preds_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Inductive preds error at {i}: {calc} vs {gold}"
        );
    }

    // 3. FTRL-Proximal on CPU and Auto
    let grad_stream_flat: Vec<f32> = tensors
        .get("grad_stream")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let final_w_gold = tensors.get("final_w_ftrl").unwrap().to_vec1::<f32>()?;
    let z_gold = tensors.get("z_accum").unwrap().to_vec1::<f32>()?;
    let n_gold = tensors.get("n_accum").unwrap().to_vec1::<f32>()?;

    let (w_calc, z_calc, n_calc) =
        t12::ftrl_proximal_cpu(&grad_stream_flat, 4, 4, 0.1, 1.0, 0.5, 0.1);
    for (i, (&calc, &gold)) in w_calc.iter().zip(final_w_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "FTRL w error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in z_calc.iter().zip(z_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "FTRL z error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in n_calc.iter().zip(n_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "FTRL n error at {i}: {calc} vs {gold}"
        );
    }

    let w_auto = t12::ftrl_threshold_auto(None, None, &z_gold, &n_gold, 4, 0.1, 1.0, 0.5, 0.1);
    for (i, (&calc, &gold)) in w_auto.iter().zip(final_w_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "FTRL w auto error at {i}: {calc} vs {gold}"
        );
    }

    // 4. Page-Hinkley CUSUM Detector on CPU
    let stream_errors = tensors.get("stream_errors").unwrap().to_vec1::<f32>()?;
    let cusum_s_gold = tensors.get("cusum_s").unwrap().to_vec1::<f32>()?;
    let ph_stat_gold = tensors.get("ph_stat").unwrap().to_vec1::<f32>()?;
    let drift_flags_gold = tensors.get("drift_flags").unwrap().to_vec1::<f32>()?;

    let (s_calc, ph_calc, d_calc) = t12::page_hinkley_cusum_cpu(&stream_errors, 0.05, 0.5, 0.10);
    for (i, (&calc, &gold)) in s_calc.iter().zip(cusum_s_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CUSUM s error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in ph_calc.iter().zip(ph_stat_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "PH stat error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in d_calc.iter().zip(drift_flags_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Drift flags error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_13_information_geometry_cpu_and_auto() -> candle_core::Result<()> {
    use sml_kernel::c01_paradigms_and_dichotomies::t13_information_geometry_solomonoff_mdl_and_maxent_free_energy as t13;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let x_anchor = tensors.get("x_anchor").unwrap().to_vec2::<f32>()?;
    let x_anchor_flat: Vec<f32> = x_anchor.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let anchor_gold = tensors.get("linear_anchor_out").unwrap().to_vec2::<f32>()?;
    let anchor_gold_flat: Vec<f32> = anchor_gold.into_iter().flatten().collect();

    let anchor_calc = t13::linear_anchor_forward_cpu(&x_anchor_flat, &weight_flat, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(anchor_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Amari Dually Flat Geometry on CPU
    let theta = tensors.get("theta").unwrap().to_vec1::<f32>()?;
    let psi_gold = tensors.get("psi").unwrap().to_vec1::<f32>()?[0];
    let eta_gold = tensors.get("eta").unwrap().to_vec1::<f32>()?;
    let phi_gold = tensors.get("phi").unwrap().to_vec1::<f32>()?[0];
    let gap_gold = tensors.get("legendre_gap").unwrap().to_vec1::<f32>()?[0];
    let fisher_gold = tensors
        .get("fisher_metric")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();

    let psi_calc = t13::log_partition_cpu(&theta);
    assert!(
        (psi_calc - psi_gold).abs() < 1e-5,
        "psi error: {psi_calc} vs {psi_gold}"
    );

    let eta_calc = t13::expectation_param_cpu(&theta);
    for (i, (&calc, &gold)) in eta_calc.iter().zip(eta_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "eta error at {i}: {calc} vs {gold}"
        );
    }

    let phi_calc = t13::dual_potential_cpu(&eta_calc);
    assert!(
        (phi_calc - phi_gold).abs() < 1e-5,
        "phi error: {phi_calc} vs {phi_gold}"
    );

    let gap_calc = t13::legendre_gap_cpu(&theta, &eta_calc);
    assert!(
        (gap_calc - gap_gold).abs() < 1e-5,
        "gap error: {gap_calc} vs {gap_gold}"
    );

    let fisher_calc = t13::fisher_rao_metric_cpu(&eta_calc);
    for (i, (&calc, &gold)) in fisher_calc.iter().zip(fisher_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Fisher error at {i}: {calc} vs {gold}"
        );
    }

    // 3. Dual Geodesics (CPU & Auto)
    let theta_1 = tensors.get("theta_1").unwrap().to_vec1::<f32>()?;
    let theta_2 = tensors.get("theta_2").unwrap().to_vec1::<f32>()?;
    let eta_1 = tensors.get("eta_1").unwrap().to_vec1::<f32>()?;
    let eta_2 = tensors.get("eta_2").unwrap().to_vec1::<f32>()?;
    let e_mid_gold = tensors.get("e_geodesic_mid").unwrap().to_vec1::<f32>()?;
    let m_mid_gold = tensors
        .get("m_geodesic_mid_eta")
        .unwrap()
        .to_vec1::<f32>()?;
    let kl_fwd_gold = tensors.get("kl_forward").unwrap().to_vec1::<f32>()?[0];
    let kl_rev_gold = tensors.get("kl_reverse").unwrap().to_vec1::<f32>()?[0];
    let bregman_gold = tensors.get("bregman_div").unwrap().to_vec1::<f32>()?[0];

    let e_mid_cpu = t13::e_geodesic_cpu(&theta_1, &theta_2, 0.5);
    for (i, (&calc, &gold)) in e_mid_cpu.iter().zip(e_mid_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "e-geodesic CPU error at {i}: {calc} vs {gold}"
        );
    }

    let e_mid_auto = pollster::block_on(t13::dual_geodesic_auto(None, &theta_1, &theta_2, 0.5));
    for (i, (&calc, &gold)) in e_mid_auto.iter().zip(e_mid_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "e-geodesic Auto error at {i}: {calc} vs {gold}"
        );
    }

    let m_mid_cpu = t13::m_geodesic_eta_cpu(&eta_1, &eta_2, 0.5);
    for (i, (&calc, &gold)) in m_mid_cpu.iter().zip(m_mid_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "m-geodesic CPU error at {i}: {calc} vs {gold}"
        );
    }

    let kl_fwd_calc = t13::kl_divergence_cpu(&eta_1, &eta_2);
    assert!(
        (kl_fwd_calc - kl_fwd_gold).abs() < 1e-5,
        "KL fwd error: {kl_fwd_calc} vs {kl_fwd_gold}"
    );

    let kl_rev_calc = t13::kl_divergence_cpu(&eta_2, &eta_1);
    assert!(
        (kl_rev_calc - kl_rev_gold).abs() < 1e-5,
        "KL rev error: {kl_rev_calc} vs {kl_rev_gold}"
    );

    let bregman_calc = t13::bregman_divergence_cpu(&theta_1, &theta_2, &eta_1);
    assert!(
        (bregman_calc - bregman_gold).abs() < 1e-5,
        "Bregman error: {bregman_calc} vs {bregman_gold}"
    );

    // 4. Shtarkov NML & Prequential MDL on CPU
    let stream_bits = tensors.get("stream_bits").unwrap().to_vec1::<f32>()?;
    let nml_gold = tensors.get("nml_log_comp").unwrap().to_vec1::<f32>()?[0];
    let preq_gold = tensors
        .get("prequential_codelength")
        .unwrap()
        .to_vec1::<f32>()?[0];

    let nml_calc = t13::shtarkov_log_complexity_cpu(8);
    assert!(
        (nml_calc - nml_gold).abs() < 1e-5,
        "NML error: {nml_calc} vs {nml_gold}"
    );

    let preq_calc = t13::prequential_codelength_cpu(&stream_bits);
    assert!(
        (preq_calc - preq_gold).abs() < 1e-5,
        "Prequential error: {preq_calc} vs {preq_gold}"
    );

    // 5. Variational Free Energy on CPU
    let q_mu = tensors.get("q_mu").unwrap().to_vec1::<f32>()?;
    let q_sigma = tensors.get("q_sigma").unwrap().to_vec1::<f32>()?;
    let p_prior_mu = tensors.get("p_prior_mu").unwrap().to_vec1::<f32>()?;
    let p_prior_sigma = tensors.get("p_prior_sigma").unwrap().to_vec1::<f32>()?;
    let obs_x = tensors.get("obs_x").unwrap().to_vec1::<f32>()?;
    let a_mat = tensors
        .get("a_mat")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let kl_qp_gold = tensors.get("kl_qp").unwrap().to_vec1::<f32>()?[0];
    let recon_gold = tensors.get("recon_loss").unwrap().to_vec1::<f32>()?[0];
    let fe_gold = tensors.get("free_energy").unwrap().to_vec1::<f32>()?[0];

    let (kl_calc, recon_calc, fe_calc) = t13::compute_variational_free_energy_cpu(
        &q_mu,
        &q_sigma,
        &p_prior_mu,
        &p_prior_sigma,
        &obs_x,
        &a_mat,
        0.2,
    );
    assert!(
        (kl_calc - kl_qp_gold).abs() < 1e-5,
        "FE KL error: {kl_calc} vs {kl_qp_gold}"
    );
    assert!(
        (recon_calc - recon_gold).abs() < 1e-5,
        "Recon error: {recon_calc} vs {recon_gold}"
    );
    assert!(
        (fe_calc - fe_gold).abs() < 1e-5,
        "FE error: {fe_calc} vs {fe_gold}"
    );

    Ok(())
}
