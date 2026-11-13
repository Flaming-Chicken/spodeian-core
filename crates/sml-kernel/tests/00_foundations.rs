use ai_kernel::{
    KernelDevice,
    c00_foundations::{
        t01_linear_neuron_and_invertible_transforms as t01,
        t02_tensors_views_strides_and_memory_layouts as t02,
        t03_autograd_mechanics_complex_wirtinger_and_custom_functions as t03,
        t04_forward_ad_and_functional_transforms_torch_func as t04,
        t05_hooks_and_saved_tensor_hooks as t05,
        t06_nn_module_internals_parametrizations_and_meta_device as t06,
        t07_feature_engineering_scalers_and_tabular_preprocessing as t07,
        t08_reproducible_preprocessing_pipelines_and_determinism as t08,
        t09_nested_ragged_and_specialized_sparse_tensors as t09,
        t10_floating_point_arithmetic_conditioning_and_numerical_stability as t10,
        t11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp as t11,
    },
};
use candle_core::{Device, safetensors::load};

mod common;

#[test]
fn test_01_linear_neuron_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/01_linear_neuron_and_invertible_transforms.safetensors",
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
        .or_else(|| tensors.get("weight"))
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors
        .get("layers.bias")
        .or_else(|| tensors.get("bias"))
        .expect("bias")
        .to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t01::linear_forward(
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

    let y_auto = t01::linear_forward(
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

    Ok(())
}

#[test]
fn test_02_tensors_views_strides_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/02_tensors_views_strides_and_memory_layouts.safetensors",
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

    let strides = t02::standard_strides(&[4, 2]);
    assert_eq!(strides, vec![2, 1]);
    assert_eq!(t02::linear_index(&[2, 1], &strides), 5);

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

    Ok(())
}

#[test]
fn test_03_autograd_mechanics_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.safetensors",
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

    Ok(())
}

#[test]
fn test_04_forward_ad_jvp_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors",
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
    let vx = tensors
        .get("layers.tangent_in")
        .expect("tangent_in")
        .to_vec2::<f32>()?;
    let vy_gold = tensors
        .get("layers.tangent_out")
        .expect("tangent_out")
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
    let vx_flat: Vec<f32> = vx.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();
    let vy_gold_flat: Vec<f32> = vy_gold.into_iter().flatten().collect();

    // CPU test
    let (y_cpu, vy_cpu) = t04::linear_jvp(
        &KernelDevice::Cpu,
        &x_flat,
        &vx_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU primals at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in vy_cpu.iter().zip(vy_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "CPU tangents at {i}: {calc} vs {gold}"
        );
    }

    // Auto test (WGPU + CPU fallback)
    let (y_auto, vy_auto) = t04::linear_jvp(
        &KernelDevice::auto(),
        &x_flat,
        &vx_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto primals at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in vy_auto.iter().zip(vy_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Auto tangents at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_05_hooks_backward_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path =
        common::get_weights_path("00_foundations/05_hooks_and_saved_tensor_hooks.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors
        .get("test_input")
        .expect("test_input")
        .to_vec2::<f32>()?;
    let y_gold = tensors
        .get("test_output")
        .expect("test_output")
        .to_vec2::<f32>()?;
    let target = tensors
        .get("layers.target")
        .expect("target")
        .to_vec2::<f32>()?;
    let golden_gw = tensors
        .get("layers.grad_weight")
        .expect("grad_weight")
        .to_vec2::<f32>()?;
    let golden_gb = tensors
        .get("layers.grad_bias")
        .expect("grad_bias")
        .to_vec1::<f32>()?;
    let golden_gx = tensors
        .get("layers.grad_input")
        .expect("grad_input")
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
    let target_flat: Vec<f32> = target.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();
    let gw_gold_flat: Vec<f32> = golden_gw.into_iter().flatten().collect();
    let gx_gold_flat: Vec<f32> = golden_gx.into_iter().flatten().collect();

    let y_pred = t05::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_pred.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Forward error at {i}: {calc} vs {gold}"
        );
    }

    let (gx, gw, gb) = t05::linear_backward_cpu(
        &x_flat,
        &y_pred,
        &target_flat,
        &w_flat,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in gx.iter().zip(gx_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-4,
            "grad_input error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in gw.iter().zip(gw_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-4,
            "grad_weight error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in gb.iter().zip(golden_gb.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-4,
            "grad_bias error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_06_cayley_orthogonal_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/06_nn_module_internals_parametrizations_and_meta_device.safetensors",
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
    let skew = tensors
        .get("layers.skew_param")
        .expect("skew_param")
        .to_vec2::<f32>()?;
    let w_gold = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let n = skew.len();
    let skew_flat: Vec<f32> = skew.into_iter().flatten().collect();
    let w_gold_flat: Vec<f32> = w_gold.into_iter().flatten().collect();

    let w_calc = t06::cayley_transform(&skew_flat, n);
    for (i, (&calc, &gold)) in w_calc.iter().zip(w_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Cayley weight error at {i}: {calc} vs {gold}"
        );
    }

    let batch = x.len();
    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_auto = t06::linear_forward(&KernelDevice::auto(), &x_flat, &w_calc, &b, batch, n, n);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Forward prediction error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_07_feature_engineering_folded_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.safetensors",
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
    let mean = tensors.get("layers.mean").expect("mean").to_vec1::<f32>()?;
    let std = tensors.get("layers.std").expect("std").to_vec1::<f32>()?;
    let w_head = tensors
        .get("layers.w_head")
        .expect("w_head")
        .to_vec2::<f32>()?;
    let b_head = tensors
        .get("layers.b_head")
        .expect("b_head")
        .to_vec1::<f32>()?;
    let w_gold = tensors
        .get("layers.weight")
        .expect("weight")
        .to_vec2::<f32>()?;
    let b_gold = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let in_dim = w_head.len();
    let out_dim = w_head[0].len();
    let w_head_flat: Vec<f32> = w_head.into_iter().flatten().collect();
    let w_gold_flat: Vec<f32> = w_gold.into_iter().flatten().collect();

    let (w_fused, b_fused) = t07::fold_weights(&mean, &std, &w_head_flat, &b_head, in_dim, out_dim);
    for (i, (&calc, &gold)) in w_fused.iter().zip(w_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Fused weight error at {i}: {calc} vs {gold}"
        );
    }
    for (i, (&calc, &gold)) in b_fused.iter().zip(b_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Fused bias error at {i}: {calc} vs {gold}"
        );
    }

    let batch = x.len();
    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_auto = t07::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_fused,
        &b_fused,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Forward prediction error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_08_determinism_kahan_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors",
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
    let kahan_mean_gold = tensors
        .get("layers.kahan_mean")
        .expect("kahan_mean")
        .to_vec1::<f32>()?;
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

    let k_sum = t08::kahan_sum_2d_cpu(&x_flat, batch, in_dim);
    for j in 0..in_dim {
        let calc_mean = k_sum[j] / (batch as f32);
        assert!(
            (calc_mean - kahan_mean_gold[j]).abs() < 1e-6,
            "Kahan mean error at col {j}"
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
            "Forward prediction error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_09_ragged_sparse_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/09_nested_ragged_and_specialized_sparse_tensors.safetensors",
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
    let pooled_gold = tensors
        .get("layers.pooled_out")
        .expect("pooled_out")
        .to_vec2::<f32>()?;
    let offsets_i64 = tensors
        .get("layers.offsets")
        .expect("offsets")
        .to_vec1::<i64>()?;
    let offsets: Vec<usize> = offsets_i64.into_iter().map(|v| v as usize).collect();

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
    let pooled_gold_flat: Vec<f32> = pooled_gold.into_iter().flatten().collect();

    let token_pred = t09::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in token_pred.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Token pred error at {i}: {calc} vs {gold}"
        );
    }

    let pooled_calc = t09::ragged_mean_pool_cpu(&token_pred, &offsets, out_dim);
    for (i, (&calc, &gold)) in pooled_calc.iter().zip(pooled_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Pooled pred error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_10_numerical_stability_welford_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.safetensors",
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
    let mean_gold = tensors
        .get("layers.welford_mean")
        .expect("welford_mean")
        .to_vec1::<f32>()?;
    let var_gold = tensors
        .get("layers.welford_var")
        .expect("welford_var")
        .to_vec1::<f32>()?;
    let lse_gold = tensors
        .get("layers.lse_ref")
        .expect("lse_ref")
        .to_vec1::<f32>()?;
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

    let (mean_calc, var_calc) = t10::welford_mean_var_cpu(&x_flat, batch, in_dim);
    for j in 0..in_dim {
        assert!(
            (mean_calc[j] - mean_gold[j]).abs() < 1e-6,
            "Welford mean error at {j}"
        );
        assert!(
            (var_calc[j] - var_gold[j]).abs() < 1e-6,
            "Welford var error at {j}"
        );
    }

    let y_pred = t10::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_pred.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Forward pred error at {i}: {calc} vs {gold}"
        );
    }

    let lse_calc = t10::log_sum_exp_stable_cpu(&y_pred, batch, out_dim);
    for (i, (&calc, &gold)) in lse_calc.iter().zip(lse_gold.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "LSE error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}

#[test]
fn test_11_spectral_hutchpp_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path(
        "00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.safetensors",
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
    let v_init = tensors
        .get("layers.v_init")
        .expect("v_init")
        .to_vec1::<f32>()?;
    let sigma_gold = tensors
        .get("layers.spectral_sigma")
        .expect("spectral_sigma")
        .to_vec1::<f32>()?[0];

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();
    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let (sigma_calc, _, _) =
        t11::power_iteration_spectral_norm_cpu(&w_flat, &v_init, in_dim, out_dim, 5);
    assert!(
        (sigma_calc - sigma_gold).abs() < 1e-5,
        "Spectral sigma error: {sigma_calc} vs {sigma_gold}"
    );

    let y_pred = t11::linear_forward(
        &KernelDevice::auto(),
        &x_flat,
        &w_flat,
        &b,
        batch,
        in_dim,
        out_dim,
    );
    for (i, (&calc, &gold)) in y_pred.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Forward pred error at {i}: {calc} vs {gold}"
        );
    }

    Ok(())
}
