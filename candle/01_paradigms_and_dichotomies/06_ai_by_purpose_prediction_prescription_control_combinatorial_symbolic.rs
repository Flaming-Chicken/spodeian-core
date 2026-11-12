//! Category 01 Topic 06: AI by Purpose: Prediction, Prescription, Control, Combinatorial, Symbolic.
//!
//! Provides native Candle implementations for:
//! 1. Cost Predictor Backbone: Maps state features to combinatorial cost vectors.
//! 2. Smart "Predict, Then Optimize" (SPO+) Loss & Exact Decision Regret (Elmachtoub & Grigas 2022).
//! 3. Causal Uplift Estimator: Separates baseline confounding from heterogeneous treatment effects (Pearl 2009).
//! 4. Symbolic Equation Learner (EQL): Discovers analytical closed-form laws via non-linear functional primitives (Martius & Lampert 2016).
//!
//! # Mathematical Invariants
//! - Regret Non-Negativity: For any predicted cost $\hat{\mathbf{c}}$ and oracle $\mathbf{c}^*$,
//!   $\mathrm{Regret}(\hat{\mathbf{c}}, \mathbf{c}^*) = \mathbf{c}^{*\top} \mathbf{w}^*(\hat{\mathbf{c}}) - \mathbf{c}^{*\top} \mathbf{w}^*(\mathbf{c}^*) \ge 0$.
//! - SPO+ Convex Upper Bound: $\mathcal{L}_{\mathrm{SPO}+}(\hat{\mathbf{c}}, \mathbf{c}^*) \ge \mathrm{Regret}(\hat{\mathbf{c}}, \mathbf{c}^*)$.
//! - Causal Identifiability: Unconfoundedness and positivity guarantee $\tau(\mathbf{x}) = \mathbb{E}[Y | X=\mathbf{x}, T=1] - \mathbb{E}[Y | X=\mathbf{x}, T=0]$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c01_paradigms_and_dichotomies::t06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic::CostPredictor;
//!
//! let device = Device::Cpu;
//! let w1 = Tensor::zeros((4, 4), candle_core::DType::F32, &device).unwrap();
//! let b1 = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
//! let w2 = Tensor::zeros((4, 4), candle_core::DType::F32, &device).unwrap();
//! let b2 = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
//!
//! let model = CostPredictor::new(w1, b1, w2, b2);
//! let x = Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap();
//! let c_pred = model.forward(&x).unwrap();
//! assert_eq!(c_pred.dims(), &[2, 4]);
//! ```

use candle_core::{Result, Tensor};

/// Numerically stable Softplus: $\ln(1 + \exp(z))$.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape();
    let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(x_flat.len());
    for val in x_flat {
        if val > 20.0 {
            out.push(val);
        } else if val < -20.0 {
            out.push(val.exp());
        } else {
            out.push((1.0f32 + val.exp()).ln());
        }
    }
    Tensor::from_vec(out, dims, x.device())
}

/// Cost Predictor Backbone.
#[derive(Debug, Clone)]
pub struct CostPredictor {
    pub w_cost1: Tensor,
    pub b_cost1: Tensor,
    pub w_cost2: Tensor,
    pub b_cost2: Tensor,
}

impl CostPredictor {
    pub fn new(w_cost1: Tensor, b_cost1: Tensor, w_cost2: Tensor, b_cost2: Tensor) -> Self {
        Self {
            w_cost1,
            b_cost1,
            w_cost2,
            b_cost2,
        }
    }

    /// Forward pass: $\mathrm{Softplus}(\mathrm{ReLU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2)$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x
            .matmul(&self.w_cost1)?
            .broadcast_add(&self.b_cost1)?
            .relu()?;
        let z = h.matmul(&self.w_cost2)?.broadcast_add(&self.b_cost2)?;
        softplus(&z)
    }
}

/// Smart "Predict, Then Optimize" (SPO+) Loss and Decision Regret Estimator.
pub struct SPOPlusEstimator;

impl SPOPlusEstimator {
    /// Exact Linear Program solver over unit simplex: $\mathbf{w}^*(\mathbf{c}) = \arg\min_{\mathbf{w} \in \Delta} \mathbf{c}^\top \mathbf{w}$.
    pub fn solve_linear_program(cost_vectors: &Tensor) -> Result<Tensor> {
        let dims = cost_vectors.dims();
        let batch_size = dims[0];
        let cost_dim = dims[1];
        let vals = cost_vectors.to_vec2::<f32>()?;

        let mut one_hot = vec![0.0f32; batch_size * cost_dim];
        for (i, row) in vals.iter().enumerate().take(batch_size) {
            let mut min_idx = 0;
            let mut min_val = f32::INFINITY;
            for (j, &c) in row.iter().enumerate().take(cost_dim) {
                if c < min_val {
                    min_val = c;
                    min_idx = j;
                }
            }
            one_hot[i * cost_dim + min_idx] = 1.0;
        }

        Tensor::from_vec(one_hot, (batch_size, cost_dim), cost_vectors.device())
    }

    /// Evaluates exact Decision Regret: $\mathbf{c}^{*\top} \mathbf{w}^*(\hat{\mathbf{c}}) - \mathbf{c}^{*\top} \mathbf{w}^*(\mathbf{c}^*)$.
    pub fn decision_regret(c_pred: &Tensor, c_star: &Tensor) -> Result<Tensor> {
        let w_pred = Self::solve_linear_program(c_pred)?;
        let w_star = Self::solve_linear_program(c_star)?;

        let cost_actual = c_star.mul(&w_pred)?.sum_keepdim(1)?.squeeze(1)?;
        let cost_oracle = c_star.mul(&w_star)?.sum_keepdim(1)?.squeeze(1)?;
        cost_actual.sub(&cost_oracle)
    }

    /// Evaluates SPO+ convex surrogate loss:
    /// $\max_i (c_i^* - 2 \hat{c}_i) + 2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}^*) - \mathbf{c}^{*\top} \mathbf{w}^*(\mathbf{c}^*)$.
    pub fn spo_plus_loss(c_pred: &Tensor, c_star: &Tensor) -> Result<Tensor> {
        let w_star = Self::solve_linear_program(c_star)?;
        let cost_oracle = c_star.mul(&w_star)?.sum_keepdim(1)?.squeeze(1)?;

        let two_c_pred = (c_pred * 2.0)?;
        let v_vec = c_star.sub(&two_c_pred)?;
        let max_term = v_vec.max_keepdim(1)?.squeeze(1)?;

        let lin_term = c_pred
            .mul(&w_star)?
            .sum_keepdim(1)?
            .squeeze(1)?
            .affine(2.0, 0.0)?
            .sub(&cost_oracle)?;

        max_term.add(&lin_term)
    }
}

/// Causal Treatment Effect & Counterfactual Outcome Estimator.
#[derive(Debug, Clone)]
pub struct CausalUpliftEstimator {
    pub w_tau: Tensor,
    pub b_tau: Tensor,
    pub w_mu0: Tensor,
    pub b_mu0: Tensor,
}

impl CausalUpliftEstimator {
    pub fn new(w_tau: Tensor, b_tau: Tensor, w_mu0: Tensor, b_mu0: Tensor) -> Self {
        Self {
            w_tau,
            b_tau,
            w_mu0,
            b_mu0,
        }
    }

    /// Predicts heterogeneous treatment effect: $\tau(\mathbf{x}) = \mathbf{x} \mathbf{W}_\tau + \mathbf{b}_\tau$.
    pub fn predict_treatment_effect(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.w_tau)?.broadcast_add(&self.b_tau)
    }

    /// Forward potential outcome: $Y(T) = \mu_0(\mathbf{x}) + T \cdot \tau(\mathbf{x})$.
    pub fn forward(&self, x: &Tensor, treatment_t: &Tensor) -> Result<Tensor> {
        let mu0 = x.matmul(&self.w_mu0)?.broadcast_add(&self.b_mu0)?;
        let tau = self.predict_treatment_effect(x)?;
        let uplift = treatment_t.mul(&tau)?;
        mu0.add(&uplift)
    }
}

/// Equation Learner (EQL) with non-linear functional primitives.
#[derive(Debug, Clone)]
pub struct SymbolicEquationLearner {
    pub w_eql: Tensor,
    pub b_eql: Tensor,
}

impl SymbolicEquationLearner {
    pub fn new(w_eql: Tensor, b_eql: Tensor) -> Self {
        Self { w_eql, b_eql }
    }

    /// Expands raw input into analytical primitives: $[x_0, \sin(x_1), \cos(x_2), x_0 \cdot x_3]$.
    pub fn expand_primitives(x: &Tensor) -> Result<Tensor> {
        let x0 = x.narrow(1, 0, 1)?;
        let x1 = x.narrow(1, 1, 1)?.sin()?;
        let x2 = x.narrow(1, 2, 1)?.cos()?;
        let x3 = x.narrow(1, 3, 1)?;
        let x0_x3 = x0.mul(&x3)?;
        Tensor::cat(&[&x0, &x1, &x2, &x0_x3], 1)
    }

    /// Forward pass: evaluated analytical equation.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let features = Self::expand_primitives(x)?;
        features.matmul(&self.w_eql)?.broadcast_add(&self.b_eql)
    }
}
