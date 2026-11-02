//! XAI Local Surrogates LIME SHAP and Perturbation Masks
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let n = b.len();
    let mut aug: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for i in 0..n {
        let mut pivot = i;
        for j in (i + 1)..n {
            if aug[j][i].abs() > aug[pivot][i].abs() {
                pivot = j;
            }
        }
        aug.swap(i, pivot);

        let diag = aug[i][i];
        if diag.abs() > 1e-12 {
            for col in i..=n {
                aug[i][col] /= diag;
            }
        }

        for j in 0..n {
            if j != i {
                let factor = aug[j][i];
                for col in i..=n {
                    aug[j][col] -= factor * aug[i][col];
                }
            }
        }
    }

    (0..n).map(|i| aug[i][n]).collect()
}

/// Closed-form weighted ridge regression for LIME feature attribution.
pub fn lime_weighted_regression(
    z_prime: &Tensor,
    y: &Tensor,
    pi: &Tensor,
    reg: f64,
) -> Result<Tensor> {
    let sqrt_pi = pi.sqrt()?.unsqueeze(1)?;
    let z_w = z_prime.broadcast_mul(&sqrt_pi)?;
    let y_w = y.mul(&pi.sqrt()?)?;

    let lhs = z_w.t()?.matmul(&z_w)?;
    let d = z_prime.dim(1)?;
    let eye = Tensor::eye(d, candle_core::DType::F32, z_prime.device())?.affine(reg, 0.0)?;
    let lhs_reg = lhs.add(&eye)?;

    let rhs = z_w.t()?.matmul(&y_w.unsqueeze(1)?)?.squeeze(1)?;

    let lhs_mat = lhs_reg.to_vec2::<f32>()?;
    let rhs_vec = rhs.flatten_all()?.to_vec1::<f32>()?;

    let sol = solve_linear_system(&lhs_mat, &rhs_vec);
    Tensor::from_vec(sol, (d,), z_prime.device())
}

fn comb(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut c = 1;
    for i in 0..k {
        c = c * (n - i) / (i + 1);
    }
    c
}

/// Exact KernelSHAP combinatorial weights for subsets in [1, M-1].
pub fn shapley_kernel_weights(coalition_sizes: &Tensor, m_features: usize) -> Result<Tensor> {
    let s_vec = coalition_sizes.flatten_all()?.to_vec1::<f32>()?;
    let mut weights = Vec::with_capacity(s_vec.len());

    for s_val in s_vec {
        let s = s_val as usize;
        if s > 0 && s < m_features {
            let binom = comb(m_features, s);
            let w = (m_features - 1) as f32 / (binom as f32 * s as f32 * (m_features - s) as f32);
            weights.push(w);
        } else {
            weights.push(1000.0);
        }
    }

    Tensor::from_vec(weights, coalition_sizes.shape(), coalition_sizes.device())
}

/// RISE feature importance attribution.
pub fn rise_feature_importance(masks: &Tensor, model_outputs: &Tensor) -> Result<Tensor> {
    let n = masks.dim(0)? as f64;
    model_outputs
        .unsqueeze(0)?
        .matmul(masks)?
        .squeeze(0)?
        .affine(1.0 / n, 0.0)
}

/// Faithfulness degradation curve under sequential feature masking.
pub fn faithfulness_degradation(y_orig: &Tensor, degraded_y: &Tensor) -> Result<Tensor> {
    y_orig.broadcast_sub(degraded_y)
}
