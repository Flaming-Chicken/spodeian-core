//! Stochastic Rounding and Master Weight Free BF16 Training
//! in candle/13_precision_and_numerics.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Stochastic Rounding to BF16 (7 mantissa bits) using uniform noise.
pub fn stochastic_rounding_bf16(x: &Tensor, noise: &Tensor) -> Result<Tensor> {
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let n_vec = noise.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(x_vec.len());

    for (&v, &n) in x_vec.iter().zip(n_vec.iter()) {
        if v == 0.0 {
            out.push(0.0f32);
            continue;
        }
        let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
        let abs_v = v.abs();
        let e = abs_v.log2().floor();
        let ulp = (2.0f32).powf(e - 7.0);
        let step_ratio = abs_v / ulp;
        let floor_ratio = step_ratio.floor();
        let frac = step_ratio - floor_ratio;
        let q_mag = if n < frac {
            (floor_ratio + 1.0) * ulp
        } else {
            floor_ratio * ulp
        };
        out.push(sign * q_mag);
    }

    Tensor::from_vec(out, x.shape(), x.device())
}

/// Deterministic Round-to-Nearest-Even (RTNE) to BF16.
pub fn deterministic_rtne_bf16(x: &Tensor) -> Result<Tensor> {
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(x_vec.len());

    for &v in &x_vec {
        if v == 0.0 {
            out.push(0.0f32);
            continue;
        }
        let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
        let abs_v = v.abs();
        let e = abs_v.log2().floor();
        let ulp = (2.0f32).powf(e - 7.0);
        let q_mag = (abs_v / ulp).round() * ulp;
        out.push(sign * q_mag);
    }

    Tensor::from_vec(out, x.shape(), x.device())
}

/// Master-weight-free SGD update directly on BF16 weights.
pub fn master_weight_free_sgd_step(
    w_bf16: &Tensor,
    grad: &Tensor,
    lr: f64,
    noise: &Tensor,
    use_stochastic_rounding: bool,
) -> Result<Tensor> {
    let scaled_grad = grad.affine(lr, 0.0)?;
    let target_w = (w_bf16 - &scaled_grad)?;
    if use_stochastic_rounding {
        stochastic_rounding_bf16(&target_w, noise)
    } else {
        deterministic_rtne_bf16(&target_w)
    }
}

/// Kahan compensated summation step: recovers discarded low-order bits.
pub fn kahan_accumulate_step(
    accum: &Tensor,
    val: &Tensor,
    comp: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let y = (val - comp)?;
    let t = (accum + &y)?;
    let comp_new = ((&t - accum)? - &y)?;
    Ok((t, comp_new))
}
