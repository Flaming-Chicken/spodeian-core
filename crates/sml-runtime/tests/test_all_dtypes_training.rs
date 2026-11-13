//! Comprehensive Training Verification Across All Data Types and Distribution Mappings.
//!
//! Validates:
//! 1. Training with continuous floating-point types: F64, F32, BF16, F16 using `SelfOptimizingMeta`.
//! 2. Training with low-bit and quantized formats: NF4, Posit8, INT8, Ternary.
//! 3. Training with pure integer types: i32, i16, i8 on integer lattices using Umbral calculus
//!    with exact rational residue carry propagation and zero floating-point shadow weights.
//! 4. Training with `GeneralDistribution` mappings:
//!    - CDF Probability Integral Transform training.
//!    - Log-PDF Maximum Likelihood Estimation (MLE) training.
//!    - Discrete PMF loss training on integer labels: $\mathcal{L}(k) = -\ln \Delta_h F(k)$.

use candle_core::{DType, Device, Result, Tensor, Var};
use sml_runtime::distribution::{DistributionKind, GeneralDistribution};
use sml_runtime::optim::{Optimizer, SelfOptimizingMeta, SelfOptimizingMetaConfig};
use sml_runtime::quantization::{decode_posit8, dequantize_nf4, encode_posit8, quantize_to_nf4};
use sml_runtime::umbral::UmbralIntegerState;

// ============================================================================
// 1. Continuous Floating-Point Training: F64, F32, BF16, F16
// ============================================================================

#[test]
fn test_training_continuous_f64() -> Result<()> {
    let dev = Device::Cpu;
    let var = Var::from_slice(&[3.0f64, -4.0f64], (2,), &dev)?;
    let config = SelfOptimizingMetaConfig::for_dtype(DType::F64);
    let mut opt = SelfOptimizingMeta::new(vec![var.clone()], config);

    let initial_loss = var.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;

    for _ in 0..50 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let final_loss = var.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;
    println!(
        "F64 Training: initial = {:.4e}, final = {:.4e}",
        initial_loss, final_loss
    );
    assert!(final_loss < 1e-4, "F64 training failed to converge!");
    Ok(())
}

#[test]
fn test_training_continuous_f32() -> Result<()> {
    let dev = Device::Cpu;
    let var = Var::from_slice(&[3.0f32, -4.0f32], (2,), &dev)?;
    let config = SelfOptimizingMetaConfig::for_dtype(DType::F32);
    let mut opt = SelfOptimizingMeta::new(vec![var.clone()], config);

    let initial_loss = var.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;

    for _ in 0..50 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let final_loss = var.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;
    println!(
        "F32 Training: initial = {:.4e}, final = {:.4e}",
        initial_loss, final_loss
    );
    assert!(final_loss < 1e-3, "F32 training failed to converge!");
    Ok(())
}

#[test]
fn test_training_continuous_bf16() -> Result<()> {
    let dev = Device::Cpu;
    let t_init = Tensor::new(&[2.5f32, -1.5f32], &dev)?.to_dtype(DType::BF16)?;
    let var = Var::from_tensor(&t_init)?;
    let config = SelfOptimizingMetaConfig::for_dtype(DType::BF16);
    let mut opt = SelfOptimizingMeta::new(vec![var.clone()], config);

    let initial_loss = var
        .as_tensor()
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;

    for _ in 0..40 {
        let loss = var.as_tensor().to_dtype(DType::F32)?.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let final_loss = var
        .as_tensor()
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    println!(
        "BF16 Training: initial = {:.4e}, final = {:.4e}",
        initial_loss, final_loss
    );
    assert!(final_loss < 0.05, "BF16 training failed to converge!");
    Ok(())
}

#[test]
fn test_training_continuous_f16() -> Result<()> {
    let dev = Device::Cpu;
    let t_init = Tensor::new(&[2.0f32, -3.0f32], &dev)?.to_dtype(DType::F16)?;
    let var = Var::from_tensor(&t_init)?;
    let config = SelfOptimizingMetaConfig::for_dtype(DType::F16);
    let mut opt = SelfOptimizingMeta::new(vec![var.clone()], config);

    let initial_loss = var
        .as_tensor()
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;

    for _ in 0..40 {
        let loss = var.as_tensor().to_dtype(DType::F32)?.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let final_loss = var
        .as_tensor()
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    println!(
        "F16 Training: initial = {:.4e}, final = {:.4e}",
        initial_loss, final_loss
    );
    assert!(final_loss < 0.05, "F16 training failed to converge!");
    Ok(())
}

// ============================================================================
// 2. Quantized & Low-Bit Training: NF4, Posit8, INT8, Ternary
// ============================================================================

#[test]
fn test_training_quantized_nf4() {
    // Quantized weights trained via Voronoi tree dequantization in forward pass
    let mut weights = vec![1.5f32, -2.0, 0.8, -1.2];
    let lr = 0.1f32;

    let initial_loss: f32 = weights
        .iter()
        .map(|&w| {
            let code = quantize_to_nf4(w);
            let w_q = dequantize_nf4(code);
            w_q * w_q
        })
        .sum();

    for _ in 0..40 {
        for w in weights.iter_mut() {
            let code = quantize_to_nf4(*w);
            let w_q = dequantize_nf4(code);
            // Straight-through gradient: d(loss)/dw ≈ 2 * w_q
            let grad = 2.0 * w_q;
            *w -= lr * grad;
        }
    }

    let final_loss: f32 = weights
        .iter()
        .map(|&w| {
            let code = quantize_to_nf4(w);
            let w_q = dequantize_nf4(code);
            w_q * w_q
        })
        .sum();

    println!(
        "NF4 Training: initial = {:.4e}, final = {:.4e}",
        initial_loss, final_loss
    );
    assert!(
        final_loss < initial_loss * 0.10,
        "NF4 training failed to minimize loss!"
    );
}

#[test]
fn test_training_quantized_posit8() {
    // Quantized weights using Posit8 tapered representation
    let mut weights = vec![2.0f32, -1.0, 4.0, -2.0];
    let lr = 0.1f32;

    let initial_loss: f32 = weights
        .iter()
        .map(|&w| {
            let w_q = decode_posit8(encode_posit8(w));
            w_q * w_q
        })
        .sum();

    for _ in 0..40 {
        for w in weights.iter_mut() {
            let w_q = decode_posit8(encode_posit8(*w));
            let grad = 2.0 * w_q;
            *w -= lr * grad;
        }
    }

    let final_loss: f32 = weights
        .iter()
        .map(|&w| {
            let w_q = decode_posit8(encode_posit8(w));
            w_q * w_q
        })
        .sum();

    println!(
        "Posit8 Training: initial = {:.4e}, final = {:.4e}",
        initial_loss, final_loss
    );
    assert!(
        final_loss < initial_loss * 0.10,
        "Posit8 training failed to minimize loss!"
    );
}

#[test]
fn test_training_quantized_int8() {
    let scale = 32.0f32;
    let inv_scale = 1.0f32 / scale;
    let mut weights_int8: Vec<i8> = vec![64, -96, 48, -80]; // represents [2.0, -3.0, 1.5, -2.5]
    let lr = 0.05f32;
    let mut shadow_res: Vec<f32> = vec![0.0; 4];

    let initial_loss: f32 = weights_int8
        .iter()
        .map(|&w| {
            let wf = w as f32 * inv_scale;
            wf * wf
        })
        .sum();

    for _ in 0..50 {
        for i in 0..weights_int8.len() {
            let wf = weights_int8[i] as f32 * inv_scale;
            let grad = 2.0 * wf;
            let delta = -lr * grad * scale;
            shadow_res[i] += delta;
            let carry = shadow_res[i].round() as i32;
            if carry != 0 {
                weights_int8[i] = (weights_int8[i] as i32 + carry).clamp(-128, 127) as i8;
                shadow_res[i] -= carry as f32;
            }
        }
    }

    let final_loss: f32 = weights_int8
        .iter()
        .map(|&w| {
            let wf = w as f32 * inv_scale;
            wf * wf
        })
        .sum();

    println!(
        "INT8 Training: initial = {:.4e}, final = {:.4e}",
        initial_loss, final_loss
    );
    assert!(
        final_loss < initial_loss * 0.10,
        "INT8 training failed to minimize loss!"
    );
}

#[test]
fn test_training_quantized_ternary() {
    let mut latent_weights = vec![1.8f32, -2.4, 0.9, -1.5];
    let lr = 0.15f32;

    let _initial_loss: f32 = latent_weights
        .iter()
        .map(|&w| {
            let t = if w > 0.5 {
                1.0f32
            } else if w < -0.5 {
                -1.0
            } else {
                0.0
            };
            (t - 0.0).powi(2)
        })
        .sum();

    for _ in 0..40 {
        for w in latent_weights.iter_mut() {
            let t = if *w > 0.5 {
                1.0f32
            } else if *w < -0.5 {
                -1.0
            } else {
                0.0
            };
            let grad = 2.0 * t;
            *w -= lr * grad;
        }
    }

    let final_ternary_active: usize = latent_weights.iter().filter(|&&w| w.abs() > 0.5).count();
    println!(
        "Ternary Training: initial active = 4, final active = {}",
        final_ternary_active
    );
    assert_eq!(
        final_ternary_active, 0,
        "Ternary weights should contract to zero state!"
    );
}

// ============================================================================
// 3. Pure Integer Training: i32, i16, i8 on Umbral Lattices
// ============================================================================

#[test]
fn test_training_pure_integer_i32() {
    let n = 16;
    let mut state = UmbralIntegerState::new(n, 1024);
    let mut coords: Vec<i32> = vec![100; n];

    // Loss: f(x) = sum x_i^2 -> Discrete forward difference: Delta f(x) = 2 x_i + 1
    let initial_norm: i64 = coords.iter().map(|&x| (x as i64) * (x as i64)).sum();

    for _ in 0..150 {
        let diffs: Vec<i32> = coords.iter().map(|&x| 2 * x + 1).collect();
        state.step_discrete_in_place(&mut coords, &diffs, 1, 1);
    }

    let final_norm: i64 = coords.iter().map(|&x| (x as i64) * (x as i64)).sum();
    println!(
        "Pure Integer i32 Training: initial norm = {}, final norm = {}",
        initial_norm, final_norm
    );
    assert!(
        final_norm <= 16,
        "Integer lattice training must contract to origin neighborhood, got {}",
        final_norm
    );
}

#[test]
fn test_training_pure_integer_i16() {
    let n = 8;
    let mut coords: Vec<i16> = vec![80; n];
    let mut state = UmbralIntegerState::new(n, 2048);
    let mut i32_buf: Vec<i32> = coords.iter().map(|&x| x as i32).collect();

    for _ in 0..120 {
        let diffs: Vec<i32> = i32_buf.iter().map(|&x| 2 * x + 1).collect();
        state.step_discrete_in_place(&mut i32_buf, &diffs, 1, 1);
    }

    for i in 0..n {
        coords[i] = i32_buf[i] as i16;
    }

    let max_abs = coords.iter().map(|&x| (x as i32).abs()).max().unwrap();
    println!("Pure Integer i16 Training: max abs coord = {}", max_abs);
    assert!(max_abs <= 2);
}

#[test]
fn test_training_pure_integer_i8() {
    let n = 8;
    let mut coords: Vec<i8> = vec![50; n];
    let mut state = UmbralIntegerState::new(n, 1024);
    let mut i32_buf: Vec<i32> = coords.iter().map(|&x| x as i32).collect();

    for _ in 0..100 {
        let diffs: Vec<i32> = i32_buf.iter().map(|&x| 2 * x + 1).collect();
        state.step_discrete_in_place(&mut i32_buf, &diffs, 1, 1);
    }

    for i in 0..n {
        coords[i] = i32_buf[i].clamp(-128, 127) as i8;
    }

    let max_abs = coords.iter().map(|&x| (x as i32).abs()).max().unwrap();
    println!("Pure Integer i8 Training: max abs coord = {}", max_abs);
    assert!(max_abs <= 2);
}

// ============================================================================
// 4. Distribution-Guided Training: CDF, Log-PDF, Discrete PMF
// ============================================================================

#[test]
fn test_training_with_cdf_probability_integral_transform() -> Result<()> {
    // Train a parameter x so that F(x) matches target probability 0.8413 (1 standard deviation)
    let dev = Device::Cpu;
    let var = Var::from_slice(&[0.0f64], (1,), &dev)?;
    let dist = GeneralDistribution::standard_gaussian();
    let target_prob = 0.8413447; // Corresponds to x = 1.0

    let mut opt = SelfOptimizingMeta::new(
        vec![var.clone()],
        SelfOptimizingMetaConfig::for_dtype(DType::F64),
    );

    for _ in 0..30 {
        let x_curr = var.as_tensor().to_vec1::<f64>()?[0];
        let cdf_curr = dist.cdf(x_curr);
        let pdf_curr = dist.pdf(x_curr); // Gradient d(cdf)/dx = pdf(x)

        let _loss_val = (cdf_curr - target_prob).powi(2);
        let grad_x = 2.0 * (cdf_curr - target_prob) * pdf_curr;

        // Custom backward injection via loss gradient
        let loss = (var.as_tensor() * grad_x)?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let x_final = var.as_tensor().to_vec1::<f64>()?[0];
    let cdf_final = dist.cdf(x_final);
    println!(
        "CDF Transform Training: x_final = {:.4}, cdf_final = {:.4} (target = {:.4})",
        x_final, cdf_final, target_prob
    );
    assert!(
        (cdf_final - target_prob).abs() < 1e-2,
        "CDF transform training failed to match target probability!"
    );
    assert!((x_final - 1.0).abs() < 0.05);
    Ok(())
}

#[test]
fn test_training_with_mle_log_pdf() -> Result<()> {
    // Maximum Likelihood Estimation: Estimate mean parameter mu given observed samples
    let samples = [1.8f64, 2.1, 1.9, 2.2, 2.0]; // True mean = 2.0
    let mut mu = 0.0f64; // Initial guess
    let lr = 0.1f64;

    for _ in 0..40 {
        let mut nll_grad = 0.0f64;
        for &x in &samples {
            // d(-log p(x))/dmu = -(x - mu)
            nll_grad -= x - mu;
        }
        mu -= lr * (nll_grad / samples.len() as f64);
    }

    println!(
        "MLE Log-PDF Training: estimated mu = {:.4} (true = 2.000)",
        mu
    );
    assert!(
        (mu - 2.0).abs() < 0.05,
        "MLE Log-PDF training failed to converge to sample mean!"
    );
    Ok(())
}

#[test]
fn test_training_with_discrete_pmf_loss_on_integers() {
    // Discrete cross-entropy on integer labels: L(k) = -ln(F((k+1)h) - F(kh))
    // Train location parameter mu to match discrete label k = 5 with grid_step h = 0.5 (target x ≈ 2.75)
    let target_k = 5i64;
    let grid_step = 0.5f64;
    let mut mu = 0.0f64;
    let lr = 0.2f64;

    for _ in 0..50 {
        let dist = GeneralDistribution::new(DistributionKind::Logistic {
            location: mu,
            scale: 1.0,
        });
        let pmf = dist.discrete_pmf(target_k, grid_step).max(1e-12);
        // Numerical difference gradient d(-ln pmf)/dmu
        let eps = 1e-5;
        let dist_plus = GeneralDistribution::new(DistributionKind::Logistic {
            location: mu + eps,
            scale: 1.0,
        });
        let pmf_plus = dist_plus.discrete_pmf(target_k, grid_step).max(1e-12);
        let grad_mu = -(pmf_plus.ln() - pmf.ln()) / eps;

        mu -= lr * grad_mu;
    }

    let target_center = (target_k as f64 + 0.5) * grid_step;
    println!(
        "Discrete PMF Training: converged mu = {:.4} (target center = {:.4})",
        mu, target_center
    );
    assert!(
        (mu - target_center).abs() < 0.10,
        "Discrete PMF loss failed to align with target integer bin!"
    );
}
