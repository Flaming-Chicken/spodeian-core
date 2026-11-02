//! Differentiable Decision Trees, NODE, and Neural Additive Models in Candle.
//!
//! Features:
//! - Neural Oblivious Decision Ensemble (NODE) forward pass with soft routing
//! - Neural Additive Model (NAM) forward pass with modular feature branches
//! - Pool-Adjacent-Violators (PAV) monotonic isotonic regression

use candle_core::{Result, Tensor};

/// Computes NODE forward pass for depth=2 oblivious trees.
pub fn node_forward(
    x: &Tensor,
    w_split: &Tensor,
    b_split: &Tensor,
    leaf_values: &Tensor,
) -> Result<Tensor> {
    let (n, _d) = x.dims2()?;
    let (n_trees, _depth, _d_split) = w_split.dims3()?;

    let mut ensemble_out = Tensor::zeros((n,), candle_core::DType::F32, x.device())?;

    for t in 0..n_trees {
        let w_t = w_split.get(t)?; // [depth=2, D]
        let b_t = b_split.get(t)?; // [depth=2]
        let leaves_t = leaf_values.get(t)?; // [4]

        let w0 = w_t.get(0)?.unsqueeze(1)?; // [D, 1]
        let w1 = w_t.get(1)?.unsqueeze(1)?; // [D, 1]
        let b0 = b_t.get(0)?; // []
        let b1 = b_t.get(1)?; // []

        let z0 = (x.matmul(&w0)?.squeeze(1)? + b0.broadcast_as((n,))?)?;
        let z1 = (x.matmul(&w1)?.squeeze(1)? + b1.broadcast_as((n,))?)?;

        let p0 = candle_nn::ops::sigmoid(&z0)?;
        let p1 = candle_nn::ops::sigmoid(&z1)?;

        let one_minus_p0 = (Tensor::ones((n,), candle_core::DType::F32, x.device())? - &p0)?;
        let one_minus_p1 = (Tensor::ones((n,), candle_core::DType::F32, x.device())? - &p1)?;

        let w_leaf0 = (&one_minus_p0 * &one_minus_p1)?;
        let w_leaf1 = (&one_minus_p0 * &p1)?;
        let w_leaf2 = (&p0 * &one_minus_p1)?;
        let w_leaf3 = (&p0 * &p1)?;

        let leaf_probs = Tensor::stack(&[&w_leaf0, &w_leaf1, &w_leaf2, &w_leaf3], 1)?; // [N, 4]
        let leaves_col = leaves_t.unsqueeze(1)?; // [4, 1]
        let tree_out = leaf_probs.matmul(&leaves_col)?.squeeze(1)?; // [N]

        ensemble_out = (&ensemble_out + &tree_out)?;
    }

    Ok(ensemble_out)
}

/// Computes Neural Additive Model forward pass: y_hat = bias0 + sum_j f_j(x_j).
pub fn nam_forward(
    x: &Tensor,
    nam_w1: &Tensor,
    nam_b1: &Tensor,
    nam_w2: &Tensor,
    nam_b2: &Tensor,
    nam_bias0: &Tensor,
) -> Result<Tensor> {
    let (n, d) = x.dims2()?;
    let mut out = nam_bias0.broadcast_as((n,))?;

    for j in 0..d {
        let x_j = x.narrow(1, j, 1)?; // [N, 1]

        let w1_j = nam_w1.get(j)?; // [H, 1]
        let b1_j = nam_b1.get(j)?; // [H]
        let w2_j = nam_w2.get(j)?; // [1, H]
        let b2_j = nam_b2.get(j)?; // [1]

        let h_linear =
            (x_j.matmul(&w1_j.t()?)? + b1_j.unsqueeze(0)?.broadcast_as((n, b1_j.dims1()?))?)?;
        let h_act = h_linear.relu()?;

        let out_j = (h_act.matmul(&w2_j.t()?)? + b2_j.broadcast_as((n, 1))?)?.squeeze(1)?;
        out = (&out + &out_j)?;
    }

    Ok(out)
}

/// Pool-Adjacent-Violators (PAV) algorithm for monotonic regression.
pub fn pav_isotonic_regression(y: &[f32]) -> Vec<f32> {
    // Each block is [start_idx, end_idx, weight, sum_val]
    let mut blocks: Vec<(usize, usize, f32, f32)> =
        (0..y.len()).map(|i| (i, i, 1.0f32, y[i])).collect();

    let mut i = 0;
    while i + 1 < blocks.len() {
        let mean_curr = blocks[i].3 / blocks[i].2;
        let mean_next = blocks[i + 1].3 / blocks[i + 1].2;

        if mean_curr > mean_next {
            blocks[i].1 = blocks[i + 1].1;
            blocks[i].2 += blocks[i + 1].2;
            blocks[i].3 += blocks[i + 1].3;
            blocks.remove(i + 1);
            if i > 0 {
                i -= 1;
            }
        } else {
            i += 1;
        }
    }

    let mut out = vec![0.0f32; y.len()];
    for b in blocks {
        let val = b.3 / b.2;
        for idx in b.0..=b.1 {
            out[idx] = val;
        }
    }
    out
}
