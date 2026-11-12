//! Non Contrastive and Masked Ssl Byol Dino Vicreg MAE JEPA
//! in candle/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper for L2 normalization along dimension 1.
fn normalize(x: &Tensor) -> Result<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(1)? + 1e-12)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// BYOL normalized prediction loss: 2 - 2 * (p / ||p||) . (z / ||z||).
pub fn byol_prediction_loss(p: &Tensor, z: &Tensor) -> Result<Tensor> {
    let p_norm = normalize(p)?;
    let z_norm = normalize(z)?;
    let cos_sim = (p_norm * z_norm)?.sum_keepdim(1)?;
    let loss = (2.0 - ((cos_sim * 2.0)?))?.mean_all()?;
    loss.reshape(&[1])
}

/// VICReg Variance-Invariance-Covariance Regularization loss.
pub fn vicreg_loss(
    za: &Tensor,
    zb: &Tensor,
    sim_coeff: f64,
    std_coeff: f64,
    cov_coeff: f64,
    gamma: f64,
    eps: f64,
) -> Result<Tensor> {
    let b = za.dim(0)?;
    let d = za.dim(1)?;

    // 1. Invariance (MSE)
    let sim_loss = (za - zb)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;

    // 2. Variance loss helper
    let var_loss = |z: &Tensor| -> Result<f64> {
        let mean = z.mean_keepdim(0)?;
        let z_centered = z.broadcast_sub(&mean)?;
        let var = ((z_centered.sqr()?.sum_keepdim(0)?) * (1.0 / (b as f64)))?;
        let std = (var + eps)?.sqrt()?;
        let std_vec = std.to_vec2::<f32>()?;
        let mut v_sum = 0.0f64;
        for j in 0..d {
            let val = gamma - (std_vec[0][j] as f64);
            if val > 0.0 {
                v_sum += val;
            }
        }
        Ok(v_sum / (d as f64))
    };

    let var_a = var_loss(za)?;
    let var_b = var_loss(zb)?;

    // 3. Covariance loss helper
    let cov_loss = |z: &Tensor| -> Result<f64> {
        let mean = z.mean_keepdim(0)?;
        let z_c = z.broadcast_sub(&mean)?;
        let cov = ((z_c.transpose(0, 1)?.matmul(&z_c)?) * (1.0 / ((b - 1) as f64)))?;
        let cov_vec = cov.to_vec2::<f32>()?;
        let mut off_sq = 0.0f64;
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    let val = cov_vec[i][j] as f64;
                    off_sq += val * val;
                }
            }
        }
        Ok(off_sq / (d as f64))
    };

    let cov_a = cov_loss(za)?;
    let cov_b = cov_loss(zb)?;

    let total = sim_coeff * sim_loss + std_coeff * (var_a + var_b) + cov_coeff * (cov_a + cov_b);
    Tensor::new(&[total as f32], za.device())
}

/// DINO cross-entropy distillation loss.
pub fn dino_loss(
    s_logits: &Tensor,
    t_logits: &Tensor,
    t_center: &Tensor,
    tau_s: f64,
    tau_t: f64,
) -> Result<Tensor> {
    let b = s_logits.dim(0)?;
    let t_scaled = (t_logits.broadcast_sub(t_center)? * (1.0 / tau_t))?;
    let p_t = candle_nn::ops::softmax(&t_scaled, 1)?;

    let s_scaled = (s_logits * (1.0 / tau_s))?;
    let log_p_s = candle_nn::ops::log_softmax(&s_scaled, 1)?;

    let loss = ((p_t * log_p_s)?.sum_all()? * (-1.0 / (b as f64)))?;
    loss.reshape(&[1])
}

/// KoLeo differential entropy regularization loss on L2-normalized representations.
pub fn koleo_loss(z: &Tensor, eps: f32) -> Result<Tensor> {
    let b = z.dim(0)?;
    let z_norm = normalize(z)?;
    let vec = z_norm.to_vec2::<f32>()?;
    let d = vec[0].len();

    let mut total_loss = 0.0f32;
    for i in 0..b {
        let mut min_d = f32::MAX;
        for j in 0..b {
            if i != j {
                let mut sum_sq = 0.0f32;
                for k in 0..d {
                    let diff = vec[i][k] - vec[j][k];
                    sum_sq += diff * diff;
                }
                let dist = sum_sq.sqrt();
                if dist < min_d {
                    min_d = dist;
                }
            }
        }
        total_loss += -(min_d + eps).ln();
    }

    Tensor::new(&[total_loss / (b as f32)], z.device())
}

/// MAE gather visible patches.
pub fn mae_patch_gather(patches: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let b = patches.dim(0)?;
    let l = patches.dim(1)?;
    let d = patches.dim(2)?;
    let k = indices.dim(1)?;

    let patches_flat = patches.flatten_all()?.to_vec1::<f32>()?;
    let indices_flat = indices.flatten_all()?.to_vec1::<i64>()?;

    let mut gathered = Vec::with_capacity(b * k * d);
    for bi in 0..b {
        for ki in 0..k {
            let idx = indices_flat[bi * k + ki] as usize;
            let offset = (bi * l + idx) * d;
            for di in 0..d {
                gathered.push(patches_flat[offset + di]);
            }
        }
    }

    Tensor::from_vec(gathered, (b, k, d), patches.device())
}

/// I-JEPA MSE prediction loss in latent representation space.
pub fn jepa_feature_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let loss = (pred - target)?.sqr()?.mean_all()?;
    loss.reshape(&[1])
}
