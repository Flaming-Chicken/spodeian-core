//! Latent Decision Gates: Highway Networks, Hard Concrete L0, Sparsemax, PonderNet Halting, and FiLM (C03 Topic 12).
//!
//! Native Rust Candle implementation of:
//! 1. Linear Anchor.
//! 2. Highway Network layer.
//! 3. Gated Linear Unit (GLU).
//! 4. Hard Concrete stretched-and-rectified discrete stochastic gates with expected L0 penalty.
//! 5. Sparsemax exact-zero simplex projection.
//! 6. PonderNet geometric halting hazard formulation with expected steps and KL prior.
//! 7. Feature-wise Linear Modulation (FiLM).

use candle_core::{Result, Tensor};

/// Linear anchor forward: Y = X @ W + b.
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

/// Highway Network forward: y = t * relu(x @ W_h + b_h) + (1 - t) * x where t = sigmoid(x @ W_t + b_t).
pub fn highway_forward(
    x: &Tensor,
    w_h: &Tensor,
    b_h: &Tensor,
    w_t: &Tensor,
    b_t: &Tensor,
) -> Result<Tensor> {
    let act_h = x.matmul(w_h)?.broadcast_add(b_h)?.relu()?;
    let act_t = candle_nn::ops::sigmoid(&x.matmul(w_t)?.broadcast_add(b_t)?)?;
    let one_minus_t = (1.0f64 - &act_t)?;
    let term1 = act_t.mul(&act_h)?;
    let term2 = one_minus_t.mul(x)?;
    term1.add(&term2)
}

/// Gated Linear Unit: z = x @ W + b, y = z[..., :d] * sigmoid(z[..., d:]).
pub fn glu_forward(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let z = x.matmul(w)?.broadcast_add(b)?;
    let dims = z.dims();
    let d2 = dims[dims.len() - 1];
    let d = d2 / 2;
    let a = z.narrow(dims.len() - 1, 0, d)?;
    let gate = z.narrow(dims.len() - 1, d, d)?;
    let sig_gate = candle_nn::ops::sigmoid(&gate)?;
    a.mul(&sig_gate)
}

/// Hard Concrete stretched-and-rectified stochastic gate with closed-form expected L0 penalty.
pub fn hard_concrete_forward(
    log_alpha: &Tensor,
    beta: f32,
    gamma: f32,
    zeta: f32,
    u: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let la_vec = log_alpha.to_vec2::<f32>()?;
    let u_vec = u.to_vec2::<f32>()?;
    let b = la_vec.len();
    let d = la_vec[0].len();

    let offset = beta * (-gamma / zeta).ln();
    let mut g_out = vec![vec![0.0f32; d]; b];
    let mut l0_out = vec![vec![0.0f32; d]; b];

    for i in 0..b {
        for j in 0..d {
            let u_val = u_vec[i][j];
            let logit_u = u_val.ln() - (1.0f32 - u_val).ln();
            let la_val = la_vec[i][j];
            let s = 1.0f32 / (1.0f32 + (-(logit_u + la_val) / beta).exp());
            let s_bar = s * (zeta - gamma) + gamma;
            g_out[i][j] = s_bar.clamp(0.0f32, 1.0f32);

            let l0_prob = 1.0f32 / (1.0f32 + (-(la_val - offset)).exp());
            l0_out[i][j] = l0_prob;
        }
    }

    let g_tensor = Tensor::new(g_out, log_alpha.device())?;
    let l0_tensor = Tensor::new(l0_out, log_alpha.device())?;
    Ok((g_tensor, l0_tensor))
}

/// Sparsemax: exact-zero projection of logits onto the probability simplex.
pub fn sparsemax_forward(z: &Tensor) -> Result<Tensor> {
    let z_vec = z.to_vec2::<f32>()?;
    let b = z_vec.len();
    let k = z_vec[0].len();

    let mut out = vec![vec![0.0f32; k]; b];
    for i in 0..b {
        let mut sorted = z_vec[i].clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = vec![0.0f32; k];
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += sorted[j];
            cumsum[j] = sum;
        }

        let mut k_star = 1usize;
        for j in 0..k {
            let bound = 1.0f32 + ((j + 1) as f32) * sorted[j];
            if bound > cumsum[j] {
                k_star = j + 1;
            }
        }

        let tau = (cumsum[k_star - 1] - 1.0f32) / (k_star as f32);
        for j in 0..k {
            out[i][j] = (z_vec[i][j] - tau).max(0.0f32);
        }
    }

    Tensor::new(out, z.device())
}

/// PonderNet geometric halting hazard formulation with expected steps and KL prior.
pub fn pondernet_halting_forward(
    lambdas: &Tensor,
    lambda_p: f32,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let lam_vec = lambdas.to_vec2::<f32>()?;
    let b = lam_vec.len();
    let n = lam_vec[0].len();

    let mut p_out = vec![vec![0.0f32; n]; b];
    let mut steps_out = vec![0.0f32; b];
    let mut kl_out = vec![0.0f32; b];

    let mut p_star = vec![0.0f32; n];
    for step in 1..=n {
        p_star[step - 1] = lambda_p * (1.0f32 - lambda_p).powi((step - 1) as i32);
    }

    for i in 0..b {
        let mut cum_unhalted = 1.0f32;
        let mut exp_steps = 0.0f32;
        let mut kl = 0.0f32;

        for step in 0..n {
            let lam = lam_vec[i][step];
            let p_step = lam * cum_unhalted;
            p_out[i][step] = p_step;
            cum_unhalted *= 1.0f32 - lam;

            let step_num = (step + 1) as f32;
            exp_steps += p_step * step_num;

            let p_clamped = p_step.max(eps);
            let p_star_clamped = p_star[step].max(eps);
            kl += p_step * (p_clamped.ln() - p_star_clamped.ln());
        }

        steps_out[i] = exp_steps;
        kl_out[i] = kl;
    }

    let p_tensor = Tensor::new(p_out, lambdas.device())?;
    let steps_tensor = Tensor::new(steps_out, lambdas.device())?;
    let kl_tensor = Tensor::new(kl_out, lambdas.device())?;
    Ok((p_tensor, steps_tensor, kl_tensor))
}

/// Feature-wise Linear Modulation (FiLM): y = gamma * x + beta.
pub fn film_forward(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let scaled = x.mul(gamma)?;
    scaled.add(beta)
}
