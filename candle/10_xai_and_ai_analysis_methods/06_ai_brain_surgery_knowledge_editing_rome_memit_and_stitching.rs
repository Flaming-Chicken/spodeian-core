//! AI Brain Surgery Knowledge Editing ROME MEMIT and Stitching
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let n = b.len();
    let mut m = vec![vec![0.0f32; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = a[i][j];
        }
        m[i][n] = b[i];
    }
    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if m[k][i].abs() > m[max_row][i].abs() {
                max_row = k;
            }
        }
        m.swap(i, max_row);
        let pivot = m[i][i];
        if pivot.abs() > 1e-12 {
            for j in i..=n {
                m[i][j] /= pivot;
            }
            for k in 0..n {
                if k != i {
                    let factor = m[k][i];
                    for j in i..=n {
                        m[k][j] -= factor * m[i][j];
                    }
                }
            }
        }
    }
    (0..n).map(|i| m[i][n]).collect()
}

/// Rank-One Model Editing (ROME — Meng et al., 2022).
pub fn rome_rank_one_edit(
    weight: &Tensor,
    key_star: &Tensor,
    val_star: &Tensor,
    cov_inv: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let v_curr = weight.matmul(&key_star.unsqueeze(1)?)?.squeeze(1)?;
    let residual = (val_star - &v_curr)?;
    let u = cov_inv.matmul(&key_star.unsqueeze(1)?)?.squeeze(1)?;

    let denom = ((key_star * &u)?.sum_all()?.to_scalar::<f32>()? + 1e-8) as f64;
    let delta_w = (residual.unsqueeze(1)?.matmul(&u.unsqueeze(0)?)? / denom)?;
    let w_new = (weight + &delta_w)?;

    Ok((delta_w, w_new))
}

/// Mass-Editing Memory in a Transformer (MEMIT — Meng et al., 2023).
pub fn memit_mass_edit(
    weight: &Tensor,
    keys: &Tensor,
    vals: &Tensor,
    cov_inv: &Tensor,
    reg: f32,
) -> Result<(Tensor, Tensor)> {
    let residuals = (vals - weight.matmul(keys)?)?;
    let u_mat = cov_inv.matmul(keys)?;
    let m = keys.dim(1)?;
    let d_out = vals.dim(0)?;

    let gram = (keys.t()?.matmul(&u_mat)?
        + (Tensor::eye(m, candle_core::DType::F32, keys.device())? * (reg as f64))?)?;
    let gram_t = gram.t()?.to_vec2::<f32>()?;
    let res_t = residuals.t()?.to_vec2::<f32>()?;

    let mut coeff_t = vec![vec![0.0f32; m]; d_out];
    for col in 0..d_out {
        let b: Vec<f32> = (0..m).map(|r| res_t[r][col]).collect();
        let sol = solve_linear_system(&gram_t, &b);
        for r in 0..m {
            coeff_t[col][r] = sol[r];
        }
    }

    let coeff_flat: Vec<f32> = coeff_t.into_iter().flatten().collect();
    let coeff = Tensor::from_vec(coeff_flat, (d_out, m), keys.device())?;
    let delta_w = coeff.matmul(&u_mat.t()?)?;
    let w_new = (weight + &delta_w)?;

    Ok((delta_w, w_new))
}

/// Model Stitching (Lenc & Vedaldi, 2015; Csiszarik et al., 2021).
pub fn model_stitching_layer(
    acts_a: &Tensor,
    acts_b: &Tensor,
    reg: f32,
) -> Result<(Tensor, Tensor)> {
    let d_a = acts_a.dim(1)?;
    let d_b = acts_b.dim(1)?;

    let lhs = (acts_a.t()?.matmul(acts_a)?
        + (Tensor::eye(d_a, candle_core::DType::F32, acts_a.device())? * (reg as f64))?)?;
    let rhs = acts_a.t()?.matmul(acts_b)?;

    let lhs_mat = lhs.to_vec2::<f32>()?;
    let rhs_mat = rhs.to_vec2::<f32>()?;

    let mut s_cols = vec![vec![0.0f32; d_a]; d_b];
    for j in 0..d_b {
        let col_b: Vec<f32> = (0..d_a).map(|i| rhs_mat[i][j]).collect();
        s_cols[j] = solve_linear_system(&lhs_mat, &col_b);
    }

    let mut s_matrix = vec![vec![0.0f32; d_b]; d_a];
    for i in 0..d_a {
        for j in 0..d_b {
            s_matrix[i][j] = s_cols[j][i];
        }
    }

    let s_flat: Vec<f32> = s_matrix.into_iter().flatten().collect();
    let s_t = Tensor::from_vec(s_flat, (d_a, d_b), acts_a.device())?;
    let b_hat = acts_a.matmul(&s_t)?;

    Ok((s_t, b_hat))
}

/// Surgical neuron pruning / knockout of specific input channels.
pub fn neuron_knockout(weight: &Tensor, knockout_mask: &Tensor) -> Result<Tensor> {
    let mask_vec = knockout_mask.to_vec1::<f32>()?;
    let keep_vec: Vec<f32> = mask_vec.into_iter().map(|m| 1.0 - m).collect();
    let keep_t = Tensor::from_vec(keep_vec, (1, knockout_mask.dim(0)?), weight.device())?;
    weight.broadcast_mul(&keep_t)
}
