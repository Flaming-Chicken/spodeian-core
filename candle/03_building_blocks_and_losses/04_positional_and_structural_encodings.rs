//! Positional and Structural Encodings: Sinusoidal, RoPE, ALiBi, and RWPE (C03 Topic 04).
//!
//! Native Rust Candle implementation of:
//! 1. Sinusoidal PE: Absolute harmonic frequency addition (Vaswani et al. 2017).
//! 2. Rotary Position Embedding (RoPE): 2D pair orthogonal rotations (Su et al. 2021).
//! 3. Attention with Linear Biases (ALiBi): Head-specific geometric distance penalty (Press et al. 2021).
//! 4. Random Walk Positional Encoding (RWPE): Landing probabilities P^k_ii (Dwivedi et al. 2022).

use candle_core::{Device, Result, Tensor};

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

/// Sinusoidal Positional Encoding forward: x + pe[0..seq_len]
pub fn sinusoidal_pe_forward(x: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let seq_len = x.dim(0)?;
    let pe_sub = pe.narrow(0, 0, seq_len)?;
    x + &pe_sub
}

/// Rotary Position Embedding (RoPE) 2D coordinate rotation.
pub fn rope_forward(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (seq_len, dim) = x.dims2()?;
    let half_dim = dim / 2;
    let x_vec = x.to_vec2::<f32>()?;
    let cos_vec = cos.to_vec2::<f32>()?;
    let sin_vec = sin.to_vec2::<f32>()?;

    let mut out = vec![vec![0.0f32; dim]; seq_len];
    for s in 0..seq_len {
        for i in 0..half_dim {
            let x1 = x_vec[s][2 * i];
            let x2 = x_vec[s][2 * i + 1];
            let c = cos_vec[s][i];
            let sn = sin_vec[s][i];
            out[s][2 * i] = x1 * c - x2 * sn;
            out[s][2 * i + 1] = x1 * sn + x2 * c;
        }
    }
    Tensor::new(out, x.device())
}

/// ALiBi attention bias matrix: -slopes[h] * |i - j|
pub fn alibi_forward(seq_len: usize, slopes: &Tensor, device: &Device) -> Result<Tensor> {
    let num_heads = slopes.dim(0)?;
    let slopes_vec = slopes.to_vec1::<f32>()?;

    let mut bias = vec![vec![vec![0.0f32; seq_len]; seq_len]; num_heads];
    for h in 0..num_heads {
        let m = slopes_vec[h];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dist = (i as f32 - j as f32).abs();
                bias[h][i][j] = -m * dist;
            }
        }
    }
    Tensor::new(bias, device)
}

/// Random Walk Positional Encoding (RWPE) landing probabilities: P^k_ii for k in 1..k_steps
pub fn rwpe_forward(adj: &Tensor, k_steps: usize) -> Result<Tensor> {
    let (num_nodes, _) = adj.dims2()?;
    let deg = adj.sum(1)?.to_vec1::<f32>()?;
    let adj_vec = adj.to_vec2::<f32>()?;

    // Transition matrix P = D^-1 A
    let mut p_walk = vec![vec![0.0f32; num_nodes]; num_nodes];
    for i in 0..num_nodes {
        let d = deg[i];
        if d > 0.0 {
            for j in 0..num_nodes {
                p_walk[i][j] = adj_vec[i][j] / d;
            }
        }
    }

    // Iterate P^k and collect diagonal
    let mut p_curr = vec![vec![0.0f32; num_nodes]; num_nodes];
    for i in 0..num_nodes {
        p_curr[i][i] = 1.0;
    }

    let mut rwpe = vec![vec![0.0f32; k_steps]; num_nodes];
    for k in 0..k_steps {
        let mut p_next = vec![vec![0.0f32; num_nodes]; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let mut sum = 0.0f32;
                for m in 0..num_nodes {
                    sum += p_curr[i][m] * p_walk[m][j];
                }
                p_next[i][j] = sum;
            }
            rwpe[i][k] = p_next[i][i];
        }
        p_curr = p_next;
    }

    Tensor::new(rwpe, adj.device())
}
