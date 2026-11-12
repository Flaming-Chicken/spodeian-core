//! Information Geometry, Solomonoff MDL, and MaxEnt Free Energy (Topic 13).
//!
//! Native Rust Candle implementation of:
//! 1. Amari's dually flat manifold: primal natural coordinates theta, dual expectation coordinates eta = grad psi,
//!    Legendre-Fenchel conjugacy psi(theta) + phi(eta) - theta^T eta = 0, Fisher-Rao Riemannian metric G(theta) = grad^2 psi,
//!    and Bregman divergence equivalence D_psi(theta_2, theta_1) = KL(p_1 || p_2).
//! 2. Dual alpha-geodesics: e-geodesic (alpha = 1, flat in theta) vs. m-geodesic (alpha = -1, flat in eta).
//! 3. Algorithmic Information & Shtarkov NML / Prequential MDL.
//! 4. Variational Free Energy: F(q) = KL(q || p_prior) + E_q[-log p(x|z)].

use candle_core::{Device, Result, Tensor};

/// Linear anchor projection module.
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

/// Amari dually flat manifold for exponential families on the simplex.
pub struct AmariDuallyFlatGeometry;

impl AmariDuallyFlatGeometry {
    pub fn log_partition(theta: &Tensor) -> Result<Tensor> {
        let max_val = theta.max(0)?;
        let shifted = theta.broadcast_sub(&max_val)?;
        let sum_exp = shifted.exp()?.sum(0)?;
        let log_sum = sum_exp.log()?;
        let res = log_sum.broadcast_add(&max_val)?;
        res.reshape(&[1])
    }

    pub fn expectation_param(theta: &Tensor) -> Result<Tensor> {
        candle_nn::ops::softmax(theta, 0)
    }

    pub fn dual_potential(eta: &Tensor) -> Result<Tensor> {
        let log_eta = eta.log()?;
        let prod = (eta * &log_eta)?;
        prod.sum_keepdim(0)
    }

    pub fn legendre_fenchel_gap(theta: &Tensor, eta: &Tensor) -> Result<Tensor> {
        let psi = Self::log_partition(theta)?;
        let phi = Self::dual_potential(eta)?;
        let inner = (theta * eta)?.sum_keepdim(0)?;
        let sum_potentials = psi.broadcast_add(&phi)?;
        let diff = sum_potentials.broadcast_sub(&inner)?;
        diff.abs()?.reshape(&[1])
    }

    pub fn fisher_rao_metric(eta: &Tensor) -> Result<Tensor> {
        let eta_vec: Vec<f32> = eta.to_vec1()?;
        let dim = eta_vec.len();
        let mut mat_data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let diag = if i == j { eta_vec[i] } else { 0.0 };
                mat_data[i * dim + j] = diag - eta_vec[i] * eta_vec[j];
            }
        }
        Tensor::from_vec(mat_data, (dim, dim), eta.device())
    }

    pub fn e_geodesic(theta_1: &Tensor, theta_2: &Tensor, t: f64) -> Result<Tensor> {
        let t1_scaled = (theta_1 * (1.0 - t))?;
        let t2_scaled = (theta_2 * t)?;
        &t1_scaled + &t2_scaled
    }

    pub fn m_geodesic_eta(eta_1: &Tensor, eta_2: &Tensor, t: f64) -> Result<Tensor> {
        let e1_scaled = (eta_1 * (1.0 - t))?;
        let e2_scaled = (eta_2 * t)?;
        &e1_scaled + &e2_scaled
    }

    pub fn kl_divergence(eta_p: &Tensor, eta_q: &Tensor) -> Result<Tensor> {
        let ratio = (eta_p / eta_q)?;
        let log_ratio = ratio.log()?;
        let prod = (eta_p * &log_ratio)?;
        prod.sum_keepdim(0)
    }

    pub fn bregman_divergence(
        theta_1: &Tensor,
        theta_2: &Tensor,
        eta_1: &Tensor,
    ) -> Result<Tensor> {
        let psi_1 = Self::log_partition(theta_1)?;
        let psi_2 = Self::log_partition(theta_2)?;
        let diff_theta = theta_2.sub(theta_1)?;
        let inner = (eta_1 * &diff_theta)?.sum_keepdim(0)?;
        let psi_diff = psi_2.sub(&psi_1)?;
        psi_diff.sub(&inner)?.reshape(&[1])
    }
}

/// Normalized Maximum Likelihood (Shtarkov Minimax Regret) and Prequential MDL.
pub struct ShtarkovNMLAndPrequentialMDL;

impl ShtarkovNMLAndPrequentialMDL {
    pub fn shtarkov_log_complexity(n: usize, device: &Device) -> Result<Tensor> {
        fn comb(n: usize, k: usize) -> f64 {
            if k > n {
                return 0.0;
            }
            let mut res = 1.0;
            for i in 1..=k {
                res = res * (n - i + 1) as f64 / i as f64;
            }
            res
        }

        let mut comp_sum = 0.0f64;
        for k in 0..=n {
            let c = comb(n, k);
            let max_lik = if k == 0 || k == n {
                1.0
            } else {
                let p = k as f64 / n as f64;
                p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
            };
            comp_sum += c * max_lik;
        }
        Tensor::new(&[comp_sum.ln() as f32], device)
    }

    pub fn prequential_codelength(stream_bits: &Tensor) -> Result<Tensor> {
        let bits: Vec<f32> = stream_bits.to_vec1()?;
        let n_seq = bits.len();
        let mut preq_loss = 0.0f64;
        let mut s_prev = 0.0f64;
        for t in 1..=n_seq {
            let bit = bits[t - 1] as f64;
            let p_pred = (s_prev + 0.5) / (t as f64);
            let prob_t = if bit == 1.0 { p_pred } else { 1.0 - p_pred };
            preq_loss -= prob_t.ln();
            s_prev += bit;
        }
        Tensor::new(&[preq_loss as f32], stream_bits.device())
    }
}

/// Variational Free Energy F(q) = KL(q || p_prior) + E_q[-log p(x|z)].
pub struct VariationalFreeEnergy {
    pub sigma_obs: f64,
}

impl VariationalFreeEnergy {
    pub fn new(sigma_obs: f64) -> Self {
        Self { sigma_obs }
    }

    pub fn compute(
        &self,
        q_mu: &Tensor,
        q_sigma: &Tensor,
        p_prior_mu: &Tensor,
        p_prior_sigma: &Tensor,
        obs_x: &Tensor,
        a_mat: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // KL(q || p) = sum(log(sig_p/sig_q) + (sig_q^2 + (mu_q - mu_p)^2)/(2 * sig_p^2) - 0.5)
        let ratio_sig = (p_prior_sigma / q_sigma)?;
        let log_ratio = ratio_sig.log()?;
        let diff_mu = q_mu.sub(p_prior_mu)?;
        let numer = ((q_sigma * q_sigma)? + (diff_mu.clone() * diff_mu)?)?;
        let denom = ((p_prior_sigma * p_prior_sigma)? * 2.0)?;
        let term2 = (numer / denom)?;
        let kl_vec = ((log_ratio + term2)? - 0.5)?;
        let kl_qp = kl_vec.sum_keepdim(0)?;

        let a_q_mu = a_mat.matmul(&q_mu.unsqueeze(1)?)?.squeeze(1)?;
        let res_mean = obs_x.sub(&a_q_mu)?;
        let sq_err_mean: f32 = (res_mean.clone() * res_mean)?.sum_all()?.to_scalar()?;

        let at_a = a_mat.t()?.matmul(a_mat)?;
        let at_a_vec: Vec<Vec<f32>> = at_a.to_vec2()?;
        let q_sig_vec: Vec<f32> = q_sigma.to_vec1()?;
        let mut tr_cov = 0.0f64;
        for i in 0..q_sig_vec.len() {
            tr_cov += (at_a_vec[i][i] as f64) * (q_sig_vec[i] as f64).powi(2);
        }

        let obs_dim = obs_x.dim(0)?;
        let sig2 = self.sigma_obs * self.sigma_obs;
        let c_const = 0.5 * (obs_dim as f64) * (2.0 * std::f64::consts::PI * sig2).ln();
        let recon_val = c_const + (sq_err_mean as f64 + tr_cov) / (2.0 * sig2);
        let recon_loss = Tensor::new(&[recon_val as f32], obs_x.device())?;
        let free_energy = (&kl_qp + &recon_loss)?;

        Ok((kl_qp, recon_loss, free_energy))
    }
}
