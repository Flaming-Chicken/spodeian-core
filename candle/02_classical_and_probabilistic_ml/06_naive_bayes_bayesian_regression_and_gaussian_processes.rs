//! Naive Bayes, Bayesian Linear Regression, and Gaussian Processes in Candle.
//!
//! Features:
//! - Gaussian Naive Bayes with log-domain log-sum-exp normalization
//! - Conjugate Bayesian Linear Regression predictive mean and variance
//! - Exact Gaussian Process RBF kernel generation and predictive equations
//! - Sparse Variational Inducing-Point diagonal variance estimation

use candle_core::{Result, Tensor};

/// Gaussian Naive Bayes classifier evaluated in log-domain.
pub struct GaussianNaiveBayes {
    pub means: Tensor,      // [C, D]
    pub vars: Tensor,       // [C, D]
    pub log_priors: Tensor, // [C]
}

impl GaussianNaiveBayes {
    pub fn new(means: Tensor, vars: Tensor, log_priors: Tensor) -> Self {
        Self {
            means,
            vars,
            log_priors,
        }
    }

    /// Evaluates normalized log-probabilities: log p(y=c | x) for each observation.
    pub fn predict_log_proba(&self, x: &Tensor) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let (c, d_m) = self.means.dims2()?;
        if d != d_m {
            candle_core::bail!(
                "Input feature dimension {} does not match mean dimension {}",
                d,
                d_m
            );
        }

        // diff_sq: [N, C, D]
        let x_exp = x.unsqueeze(1)?.broadcast_as((n, c, d))?;
        let means_exp = self.means.unsqueeze(0)?.broadcast_as((n, c, d))?;
        let vars_exp = self.vars.unsqueeze(0)?.broadcast_as((n, c, d))?;

        let diff = (x_exp - means_exp)?;
        let diff_sq = diff.sqr()?;

        // 2 * pi * vars
        let two_pi = (2.0 * std::f64::consts::PI) as f32;
        let scale = (vars_exp.affine(two_pi as f64, 0.0))?.log()?;
        let scaled_sq = diff_sq.div(&vars_exp)?;
        let sum_terms = (scale + scaled_sq)?;
        let log_gauss = sum_terms.affine(-0.5, 0.0)?; // [N, C, D]

        // Sum across feature dimension: [N, C]
        let log_lik = log_gauss.sum(2)?;
        let priors_exp = self.log_priors.unsqueeze(0)?.broadcast_as((n, c))?;
        let log_joint = (priors_exp + log_lik)?;

        // Log-sum-exp across classes C: [N, 1]
        let max_val = log_joint.max_keepdim(1)?;
        let shifted = (log_joint.sub(&max_val.broadcast_as((n, c))?))?;
        let sum_exp = shifted.exp()?.sum_keepdim(1)?;
        let lse = (sum_exp.log()? + max_val)?;

        // Normalized log probabilities: [N, C]
        log_joint.sub(&lse.broadcast_as((n, c))?)
    }
}

/// Computes RBF / Squared Exponential Kernel matrix: K_{ij} = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / l^2)
pub fn rbf_kernel(x1: &Tensor, x2: &Tensor, lengthscale: f64, sigma_f: f64) -> Result<Tensor> {
    let (n1, _d) = x1.dims2()?;
    let (n2, _) = x2.dims2()?;

    // ||x1||^2: [N1, 1]
    let x1_sq = x1.sqr()?.sum_keepdim(1)?;
    // ||x2||^2: [1, N2]
    let x2_sq = x2.sqr()?.sum_keepdim(1)?.t()?;

    // 2 * x1 @ x2^T: [N1, N2]
    let x1_x2_t = x1.matmul(&x2.t()?)?;
    let two_x1_x2_t = x1_x2_t.affine(2.0, 0.0)?;

    let dist_sq = (x1_sq.broadcast_as((n1, n2))? + x2_sq.broadcast_as((n1, n2))?)?
        .sub(&two_x1_x2_t)?
        .clamp(0.0f32, f32::MAX)?;

    let inv_two_l_sq = -0.5 / (lengthscale * lengthscale);
    let exponent = dist_sq.affine(inv_two_l_sq, 0.0)?;
    let exp_val = exponent.exp()?;
    exp_val.affine(sigma_f * sigma_f, 0.0)
}

/// Conjugate Bayesian Linear Regression predictive distribution.
pub fn bayesian_linear_regression_predict(
    x_star: &Tensor,
    m_n: &Tensor,
    s_n: &Tensor,
    beta: f64,
) -> Result<(Tensor, Tensor)> {
    // Predictive mean: mu_* = X_* m_N
    let m_col = if m_n.rank() == 1 {
        m_n.unsqueeze(1)?
    } else {
        m_n.clone()
    };
    let pred_mean = x_star.matmul(&m_col)?.squeeze(1)?;

    // Predictive variance: var_* = 1/beta + diag(X_* S_N X_*^T) = 1/beta + sum((X_* S_N) * X_*, dim=1)
    let x_sn = x_star.matmul(s_n)?;
    let quad = (x_sn * x_star)?.sum_keepdim(1)?.squeeze(1)?;
    let pred_var = quad.affine(1.0, 1.0 / beta)?;

    Ok((pred_mean, pred_var))
}

/// Gaussian Process predictive mean and variance given dual weights alpha and solve vector v.
pub fn gp_predict_with_alpha_and_v(
    k_star: &Tensor,
    alpha: &Tensor,
    v: &Tensor,
    sigma_f: f64,
    noise_var: f64,
) -> Result<(Tensor, Tensor)> {
    // Mean: K_* alpha: [N_*]
    let a_col = if alpha.rank() == 1 {
        alpha.unsqueeze(1)?
    } else {
        alpha.clone()
    };
    let pred_mean = k_star.matmul(&a_col)?.squeeze(1)?;

    // Variance: sigma_f^2 + noise_var - sum(v^2, dim=0): [N_*]
    let v_sq_sum = v.sqr()?.sum_keepdim(0)?.squeeze(0)?;
    let base_var = sigma_f * sigma_f + noise_var;
    let pred_var = v_sq_sum.affine(-1.0, base_var)?.clamp(1e-8f32, f32::MAX)?;

    Ok((pred_mean, pred_var))
}

/// Sparse Inducing Point (SVGP) diagonal covariance estimation: diag(A^T A) = sum(A^2, dim=0)
pub fn compute_svgp_inducing_diag(a: &Tensor) -> Result<Tensor> {
    a.sqr()?.sum_keepdim(0)?.squeeze(0)
}
