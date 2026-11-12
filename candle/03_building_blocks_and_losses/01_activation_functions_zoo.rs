//! Activation Functions Zoo: Rectified, Bounded, Gated, Periodic, and Self-Normalizing (C03 Topic 01).
//!
//! Native Rust Candle implementation of:
//! 1. Gated Feed-Forward: SwiGLU and GeGLU with 8/3 hidden scaling (Shazeer 2020).
//! 2. Periodic Activation: Snake x + (1 - cos(2ax)) / (2a) (Ziyin et al. 2020).
//! 3. Self-Normalizing: SELU fixed-point preservation (Klambauer et al. 2017).
//! 4. Rectified and Smooth: ReLU, LeakyReLU, and SiLU.

use candle_core::{Result, Tensor};

/// Linear anchor forward.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// SwiGLU: (silu(x @ w_gate) * (x @ w_up)) @ w_down
pub fn swiglu_forward(
    x: &Tensor,
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
) -> Result<Tensor> {
    let gate_proj = x.matmul(w_gate)?;
    let gate_act = candle_nn::ops::silu(&gate_proj)?;
    let up_proj = x.matmul(w_up)?;
    let gated = (&gate_act * &up_proj)?;
    gated.matmul(w_down)
}

/// GeGLU: (gelu(x @ w_gate) * (x @ w_up)) @ w_down
pub fn geglu_forward(
    x: &Tensor,
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
) -> Result<Tensor> {
    let gate_proj = x.matmul(w_gate)?;
    let gate_act = gate_proj.gelu_erf()?;
    let up_proj = x.matmul(w_up)?;
    let gated = (&gate_act * &up_proj)?;
    gated.matmul(w_down)
}

/// Periodic Snake activation: x + (1 - cos(2*a*x)) / (2*a)
pub fn snake_forward(x: &Tensor, a: f64) -> Result<Tensor> {
    let two_a = 2.0 * a;
    let cos_term = (x * two_a)?.cos()?;
    let ones = Tensor::ones_like(&cos_term)?;
    let numer = (&ones - &cos_term)?;
    let periodic = (numer / two_a)?;
    x + &periodic
}

/// Self-normalizing SELU activation: where(x > 0, scale * x, scale * alpha * (exp(x) - 1))
pub fn selu_forward(x: &Tensor, scale: f64, alpha: f64) -> Result<Tensor> {
    let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let out_vec: Vec<f32> = x_vec
        .into_iter()
        .map(|v| {
            if v > 0.0 {
                (scale * v as f64) as f32
            } else {
                (scale * alpha * ((v as f64).exp() - 1.0)) as f32
            }
        })
        .collect();
    Tensor::from_vec(out_vec, x.shape(), x.device())
}

/// ReLU: max(0, x)
pub fn relu_forward(x: &Tensor) -> Result<Tensor> {
    x.maximum(&Tensor::zeros_like(x)?)
}

/// LeakyReLU: where(x > 0, x, negative_slope * x)
pub fn leaky_relu_forward(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let out_vec: Vec<f32> = x_vec
        .into_iter()
        .map(|v| {
            if v > 0.0 {
                v
            } else {
                (v as f64 * negative_slope) as f32
            }
        })
        .collect();
    Tensor::from_vec(out_vec, x.shape(), x.device())
}

/// SiLU / Swish: x * sigmoid(x)
pub fn silu_forward(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
}
