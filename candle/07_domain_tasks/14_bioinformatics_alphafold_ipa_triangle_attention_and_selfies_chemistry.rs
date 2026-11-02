//! Bioinformatics Alphafold Ipa Triangle Attention and Selfies Chemistry
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// AlphaFold Triangle Multiplicative Update (Outgoing Edges).
pub fn triangle_multiplicative_update(
    z: &Tensor,
    w_a: &Tensor,
    b_a: &Tensor,
    w_b: &Tensor,
    b_b: &Tensor,
    w_g: &Tensor,
    b_g: &Tensor,
    w_o: &Tensor,
    b_o: &Tensor,
) -> Result<Tensor> {
    let (b_sz, n, _, c) = z.dims4()?;
    let z_flat = z.reshape((b_sz * n * n, c))?;

    let a = z_flat
        .matmul(w_a)?
        .broadcast_add(b_a)?
        .reshape((b_sz, n, n, c))?;
    let b = z_flat
        .matmul(w_b)?
        .broadcast_add(b_b)?
        .reshape((b_sz, n, n, c))?;
    let g = candle_nn::ops::sigmoid(&(z_flat.matmul(w_g)?.broadcast_add(b_g)?))?
        .reshape((b_sz, n, n, c))?;

    // Outgoing sum: S_{i, j, c} = sum_k a_{i, k, c} * b_{j, k, c}
    let mut s_acc = Tensor::zeros((b_sz, n, n, c), z.dtype(), z.device())?;
    for k in 0..n {
        let a_k = a.narrow(2, k, 1)?; // [B, N, 1, C]
        let b_k = b.narrow(2, k, 1)?.transpose(1, 2)?; // [B, 1, N, C]
        let term_k = a_k.broadcast_mul(&b_k)?; // [B, N, N, C]
        s_acc = (s_acc + term_k)?;
    }

    let gated = (g * s_acc)?;
    let gated_flat = gated.reshape((b_sz * n * n, c))?;
    let out = gated_flat
        .matmul(w_o)?
        .broadcast_add(b_o)?
        .reshape((b_sz, n, n, c))?;
    Ok(out)
}

/// AlphaFold Frame Aligned Point Error (FAPE) Loss.
pub fn fape_loss(
    x: &Tensor,
    y: &Tensor,
    r_x: &Tensor,
    t_x: &Tensor,
    r_y: &Tensor,
    t_y: &Tensor,
    d_clamp: f64,
) -> Result<Tensor> {
    let (b, n, _) = x.dims3()?;
    let x_vec = x.to_vec3::<f32>()?;
    let y_vec = y.to_vec3::<f32>()?;
    let rx_flat = r_x.flatten_all()?.to_vec1::<f32>()?;
    let ry_flat = r_y.flatten_all()?.to_vec1::<f32>()?;
    let tx_vec = t_x.to_vec3::<f32>()?;
    let ty_vec = t_y.to_vec3::<f32>()?;

    let mut total_loss = 0.0f32;
    let count = (b * n * n) as f32;

    for bi in 0..b {
        for i in 0..n {
            for j in 0..n {
                let mut lx = [0.0f32; 3];
                let mut ly = [0.0f32; 3];

                for a in 0..3 {
                    for c in 0..3 {
                        let rx_val = rx_flat[(bi * n + i) * 9 + a * 3 + c];
                        let ry_val = ry_flat[(bi * n + i) * 9 + a * 3 + c];

                        let dx = x_vec[bi][j][c] - tx_vec[bi][i][c];
                        lx[a] += dx * rx_val;

                        let dy = y_vec[bi][j][c] - ty_vec[bi][i][c];
                        ly[a] += dy * ry_val;
                    }
                }

                let d_sq = (lx[0] - ly[0]).powi(2)
                    + (lx[1] - ly[1]).powi(2)
                    + (lx[2] - ly[2]).powi(2)
                    + 1e-12;
                let d = (d_sq as f32).sqrt().min(d_clamp as f32);
                total_loss += d;
            }
        }
    }

    Tensor::new(total_loss / count, x.device())
}

/// AlphaFold Invariant Point Attention (IPA) 3D Point Distance Penalty.
pub fn ipa_point_distance_penalty(
    q_pts: &Tensor,
    k_pts: &Tensor,
    gamma: &Tensor,
) -> Result<Tensor> {
    let diff = q_pts.unsqueeze(2)?.broadcast_sub(&k_pts.unsqueeze(1)?)?;
    let sq_dist = diff
        .sqr()?
        .sum_keepdim(5)?
        .sum_keepdim(4)?
        .squeeze(5)?
        .squeeze(4)?;
    let gamma_exp = gamma.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?;
    sq_dist.broadcast_mul(&gamma_exp)? * (-0.5)
}

/// Molecular chemistry valence penalty loss.
pub fn molecular_valence_loss(bond_logits: &Tensor, max_valence: &Tensor) -> Result<Tensor> {
    let (_b, n, _) = bond_logits.dims3()?;
    let probs = candle_nn::ops::sigmoid(bond_logits)?;

    let mut mask_vec = vec![1.0f32; n * n];
    for i in 0..n {
        mask_vec[i * n + i] = 0.0f32;
    }
    let mask = Tensor::new(mask_vec, bond_logits.device())?.reshape((1, n, n))?;
    let masked_probs = probs.broadcast_mul(&mask)?;
    let valences = masked_probs.sum(2)?;
    let excess = (valences - max_valence)?.relu()?;
    excess.sqr()?.sum(1)?.mean_all()
}
