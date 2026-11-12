//! Profiling Roofline Analysis CUDA Graphs and Fused Optimizers
//! in candle/12_compilation_and_performance.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Evaluates Williams Roofline Model metrics:
/// Returns (arithmetic_intensity, ridge_point, attainable_tflops)
pub fn roofline_metrics(
    flops: f64,
    bytes_transferred: f64,
    peak_tflops: f64,
    peak_bandwidth_tbs: f64,
) -> (f64, f64, f64) {
    let intensity = flops / bytes_transferred.max(1.0);
    let ridge_point = peak_tflops / peak_bandwidth_tbs;
    let attainable_tflops = peak_tflops.min(intensity * peak_bandwidth_tbs);
    (intensity, ridge_point, attainable_tflops)
}

/// Fused AdamW optimizer update step.
pub fn fused_adamw_step(
    w: &Tensor,
    g: &Tensor,
    m: &Tensor,
    v: &Tensor,
    step: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let w_vec = w.to_vec2::<f32>()?;
    let g_vec = g.to_vec2::<f32>()?;
    let m_vec = m.to_vec2::<f32>()?;
    let v_vec = v.to_vec2::<f32>()?;

    let b = w_vec.len();
    let d = w_vec[0].len();

    let t = step as f32;
    let bias_corr1 = 1.0f32 - beta1.powf(t);
    let bias_corr2 = 1.0f32 - beta2.powf(t);

    let mut out_w = vec![vec![0.0f32; d]; b];
    let mut out_m = vec![vec![0.0f32; d]; b];
    let mut out_v = vec![vec![0.0f32; d]; b];

    for i in 0..b {
        for j in 0..d {
            let gv = g_vec[i][j];
            let mt = beta1 * m_vec[i][j] + (1.0f32 - beta1) * gv;
            let vt = beta2 * v_vec[i][j] + (1.0f32 - beta2) * (gv * gv);

            let m_hat = mt / bias_corr1;
            let v_hat = vt / bias_corr2;

            let w_decayed = w_vec[i][j] * (1.0f32 - lr * weight_decay);
            let wt = w_decayed - lr * (m_hat / (v_hat.sqrt() + eps));

            out_w[i][j] = wt;
            out_m[i][j] = mt;
            out_v[i][j] = vt;
        }
    }

    let dev = w.device();
    Ok((
        Tensor::new(out_w, dev)?,
        Tensor::new(out_m, dev)?,
        Tensor::new(out_v, dev)?,
    ))
}
