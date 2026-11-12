//! Candle implementation for Category 01 Topic 11:
//! Statistical Inference: Bayesian, Frequentist, Likelihoodist, MDL, and PAC-Bayes.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

/// Frequentist estimator: Gaussian mean MLE, Fisher Information, Cramér-Rao bound, and LRT statistic.
pub fn frequentist_fisher_cramer_rao(
    x_data: &Tensor,
    theta_0: &Tensor,
    sigma2_noise: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dims = x_data.dims();
    let num_obs = dims[0];
    let obs_dim = dims[1];
    let device = x_data.device();

    let theta_mle = x_data.mean(0)?;
    let fisher_val = (num_obs as f64 / sigma2_noise) as f32;
    let crlb_val = (sigma2_noise / num_obs as f64) as f32;

    let fisher_diag = Tensor::full(fisher_val, (obs_dim,), device)?;
    let cramer_rao = Tensor::full(crlb_val, (obs_dim,), device)?;

    let diff = (&theta_mle - theta_0)?;
    let diff_sq_sum = diff.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let lrt_val = (num_obs as f64 / sigma2_noise * diff_sq_sum) as f32;
    let lrt_stat = Tensor::new(lrt_val, device)?;

    Ok((theta_mle, fisher_diag, cramer_rao, lrt_stat))
}

/// Bayesian conjugate Gaussian update and Laplace marginal likelihood Occam factor.
pub fn bayesian_conjugate_laplace(
    x_data: &Tensor,
    mu_0: &Tensor,
    sigma2_0: f64,
    sigma2_noise: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dims = x_data.dims();
    let num_obs = dims[0];
    let obs_dim = dims[1];
    let device = x_data.device();

    let theta_mle = x_data.mean(0)?;

    let prec_0 = 1.0 / sigma2_0;
    let prec_lik = num_obs as f64 / sigma2_noise;
    let prec_post = prec_0 + prec_lik;
    let var_post = 1.0 / prec_post;
    let sigma_post = var_post.sqrt() as f32;

    let weighted_m0 = (mu_0 * prec_0)?;
    let weighted_mle = (&theta_mle * prec_lik)?;
    let mu_post = ((&weighted_m0 + &weighted_mle)? * var_post)?;
    let sigma_post_vec = Tensor::full(sigma_post, (obs_dim,), device)?;

    let log_det_h = obs_dim as f64 * prec_post.ln();
    let res = x_data.broadcast_sub(&mu_post)?;
    let res_sq_sum = res.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;

    let log_lik_map =
        -0.5 * num_obs as f64 * obs_dim as f64 * (2.0 * std::f64::consts::PI * sigma2_noise).ln()
            - 0.5 * res_sq_sum / sigma2_noise;

    let prior_diff = (&mu_post - mu_0)?;
    let prior_sq_sum = prior_diff.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let log_prior_map = -0.5 * obs_dim as f64 * (2.0 * std::f64::consts::PI * sigma2_0).ln()
        - 0.5 * prior_sq_sum / sigma2_0;

    let occam_penalty = -0.5 * log_det_h + 0.5 * obs_dim as f64 * (2.0 * std::f64::consts::PI).ln();
    let laplace_log_marginal = log_lik_map + log_prior_map + occam_penalty;

    let laplace_tensor = Tensor::new(laplace_log_marginal as f32, device)?;
    let occam_tensor = Tensor::new(occam_penalty as f32, device)?;

    Ok((mu_post, sigma_post_vec, laplace_tensor, occam_tensor))
}

/// James-Stein shrinkage estimator demonstrating Stein's paradox for K >= 3.
pub fn james_stein_estimator(y_obs: &Tensor, sigma2: f64) -> Result<(Tensor, Tensor, Tensor)> {
    let num_groups = y_obs.dims()[0];
    let norm_sq = y_obs.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;

    let c_shrink = 1.0 - ((num_groups - 2) as f64 * sigma2) / norm_sq;
    let c_shrink_pos = c_shrink.max(0.0);

    let theta_js = (y_obs * c_shrink)?;
    let theta_js_pos = (y_obs * c_shrink_pos)?;
    let c_tensor = Tensor::new(c_shrink as f32, y_obs.device())?;

    Ok((theta_js, theta_js_pos, c_tensor))
}

/// McAllester PAC-Bayes generalization bound for stochastic hypothesis spaces.
pub fn pac_bayes_bound(
    mu_q: &Tensor,
    sigma_q: &Tensor,
    emp_risk: &Tensor,
    num_obs: usize,
    sigma_p: f64,
    delta: f64,
) -> Result<(Tensor, Tensor)> {
    let device = mu_q.device();
    let mq_vals = mu_q.to_vec1::<f32>()?;
    let sq_vals = sigma_q.to_vec1::<f32>()?;
    let er_val = emp_risk.to_scalar::<f32>()? as f64;

    let mut kl_sum = 0.0f64;
    for (m, s) in mq_vals.iter().zip(sq_vals.iter()) {
        let m_d = *m as f64;
        let s_d = *s as f64;
        let s_sq = s_d * s_d;
        let p_sq = sigma_p * sigma_p;
        kl_sum += (m_d * m_d + s_sq) / p_sq - 1.0 - (s_sq / p_sq).ln();
    }
    let kl_div = 0.5 * kl_sum;
    let complexity_term =
        ((kl_div + (2.0 * (num_obs as f64).sqrt() / delta).ln()) / (2.0 * num_obs as f64)).sqrt();
    let pac_bound = er_val + complexity_term;

    let kl_tensor = Tensor::new(kl_div as f32, device)?;
    let pac_tensor = Tensor::new(pac_bound as f32, device)?;

    Ok((kl_tensor, pac_tensor))
}

/// Anytime-valid test martingale with Ville's inequality threshold.
pub fn anytime_valid_martingale(e_values: &Tensor, alpha: f64) -> Result<(Tensor, Tensor)> {
    let device = e_values.device();
    let vals = e_values.to_vec1::<f32>()?;
    let mut prod = 1.0f32;
    let mut cumprod = Vec::with_capacity(vals.len());
    for v in vals {
        prod *= v;
        cumprod.push(prod);
    }
    let martingale = Tensor::from_vec(cumprod, (e_values.dims()[0],), device)?;
    let threshold = Tensor::new((1.0 / alpha) as f32, device)?;
    Ok((martingale, threshold))
}

/// Top-level module encapsulating linear anchor.
pub struct StatisticalInferenceModule {
    pub linear: Linear,
}

impl StatisticalInferenceModule {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self {
            linear: Linear::new(weight, Some(bias)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}
