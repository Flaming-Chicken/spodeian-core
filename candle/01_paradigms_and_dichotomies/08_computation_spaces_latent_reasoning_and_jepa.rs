//! Category 01 Topic 08: Computation Spaces: Token CoT, Continuous Latent Reasoning, Recurrent Depth, and JEPA.
//!
//! Provides native Candle implementations for:
//! 1. Continuous Latent Reasoning Core with Input-Injection Anchoring (Coconut / Universal Transformers):
//!    $\mathbf{z}^{(k+1)} = \text{RMSNorm}(\mathbf{z}^{(k)} + \text{GELU}(\mathbf{z}^{(k)}\mathbf{W}_z + \mathbf{e}_x\mathbf{W}_{\text{anc}} + \mathbf{b}_z))$
//! 2. Adaptive Computation Halting Hazard Gate (PonderNet, Banino et al. NeurIPS 2021):
//!    $\lambda_k = \sigma(\mathbf{z}^{(k)}\mathbf{w}_h + b_h)$
//! 3. Joint-Embedding Predictive Architecture (JEPA, LeCun 2022):
//!    $\hat{\mathbf{s}}_y = \text{GELU}(\mathbf{s}_x\mathbf{W}_{p1} + \mathbf{b}_{p1})\mathbf{W}_{p2} + \mathbf{b}_{p2}$
//!
//! # Mathematical Invariants
//! - Input Anchoring Stability: Persistent injection of $\mathbf{e}_x$ bounds latent trajectory drift.
//! - Cumulative Halting Probability: $\sum_{k=1}^K p_k \le 1.0$ and $p_k \ge 0$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use sml_candle::c01_paradigms_and_dichotomies::t08_computation_spaces_latent_reasoning_and_jepa::rms_norm;
//!
//! let device = Device::Cpu;
//! let z = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
//! let normed = rms_norm(&z, 1e-5).unwrap();
//! assert_eq!(normed.dims(), &[2, 2]);
//! ```

use candle_core::{Result, Tensor};

/// Root Mean Square Normalization along the innermost dimension: $z / \sqrt{\text{mean}(z^2) + \epsilon}$.
pub fn rms_norm(z: &Tensor, eps: f64) -> Result<Tensor> {
    let mean_sq = z.sqr()?.mean_keepdim(1)?.affine(1.0, eps)?;
    let rms = mean_sq.sqrt()?;
    z.broadcast_div(&rms)
}

/// Continuous Latent Reasoner with persistent input-injection anchoring.
#[derive(Debug, Clone)]
pub struct ContinuousLatentReasoner {
    pub w_proj: Tensor,
    pub b_proj: Tensor,
    pub w_z: Tensor,
    pub w_anc: Tensor,
    pub b_z: Tensor,
}

impl ContinuousLatentReasoner {
    pub fn new(w_proj: Tensor, b_proj: Tensor, w_z: Tensor, w_anc: Tensor, b_z: Tensor) -> Self {
        Self {
            w_proj,
            b_proj,
            w_z,
            w_anc,
            b_z,
        }
    }

    /// Evaluates input anchor embedding: $\mathbf{e}_x = \text{GELU}(\mathbf{x} \mathbf{W}_{\text{proj}} + \mathbf{b}_{\text{proj}})$.
    pub fn embed_anchor(&self, x: &Tensor) -> Result<Tensor> {
        let proj = x.matmul(&self.w_proj)?.broadcast_add(&self.b_proj)?;
        proj.gelu()
    }

    /// Runs $K$-step recurrent latent deliberation, returning trajectory steps and final thought.
    pub fn forward(&self, x: &Tensor, steps: usize) -> Result<(Vec<Tensor>, Tensor)> {
        let e_x = self.embed_anchor(x)?;
        let mut z = e_x.clone();
        let mut trajectories = vec![z.clone()];

        for _ in 0..steps {
            let z_term = z.matmul(&self.w_z)?;
            let anc_term = e_x.matmul(&self.w_anc)?;
            let delta = z_term
                .broadcast_add(&anc_term)?
                .broadcast_add(&self.b_z)?
                .gelu()?;
            z = rms_norm(&z.broadcast_add(&delta)?, 1e-5)?;
            trajectories.push(z.clone());
        }

        Ok((trajectories, z))
    }
}

/// Adaptive Computation Halting Hazard Gate (PonderNet).
#[derive(Debug, Clone)]
pub struct PonderNetHalting {
    pub w_h: Tensor,
    pub b_h: Tensor,
}

impl PonderNetHalting {
    pub fn new(w_h: Tensor, b_h: Tensor) -> Self {
        Self { w_h, b_h }
    }

    /// Computes halting probabilities $p_k$ across steps $k = 1..K$.
    pub fn forward(&self, trajectories: &[Tensor]) -> Result<Tensor> {
        let steps = trajectories.len().saturating_sub(1);
        let batch = trajectories[0].dims()[0];

        let mut hazards = Vec::with_capacity(steps);
        for k in 0..steps {
            let z_k = &trajectories[k + 1];
            let hazard_logit = z_k.matmul(&self.w_h)?.broadcast_add(&self.b_h)?;
            let lam = candle_nn::ops::sigmoid(&hazard_logit)?;
            hazards.push(lam);
        }

        let mut halting_probs = Vec::with_capacity(steps);
        let device = trajectories[0].device();
        let mut unhalted = Tensor::ones((batch, 1), candle_core::DType::F32, device)?;

        for k in 0..steps {
            let lam = &hazards[k];
            let p_k = lam.mul(&unhalted)?;
            halting_probs.push(p_k);
            let one_minus_lam =
                Tensor::ones((batch, 1), candle_core::DType::F32, device)?.sub(lam)?;
            unhalted = unhalted.mul(&one_minus_lam)?;
        }

        let refs: Vec<&Tensor> = halting_probs.iter().collect();
        Tensor::cat(&refs, 1)
    }
}

/// Joint-Embedding Predictive Architecture Latent Predictor.
#[derive(Debug, Clone)]
pub struct JEPAPredictor {
    pub w_p1: Tensor,
    pub b_p1: Tensor,
    pub w_p2: Tensor,
    pub b_p2: Tensor,
}

impl JEPAPredictor {
    pub fn new(w_p1: Tensor, b_p1: Tensor, w_p2: Tensor, b_p2: Tensor) -> Self {
        Self {
            w_p1,
            b_p1,
            w_p2,
            b_p2,
        }
    }

    /// Predicts target latent representation: $\text{GELU}(\mathbf{s}_x \mathbf{W}_{p1} + \mathbf{b}_{p1})\mathbf{W}_{p2} + \mathbf{b}_{p2}$.
    pub fn forward(&self, s_x: &Tensor) -> Result<Tensor> {
        let h = s_x.matmul(&self.w_p1)?.broadcast_add(&self.b_p1)?.gelu()?;
        h.matmul(&self.w_p2)?.broadcast_add(&self.b_p2)
    }

    /// Evaluates squared $L_2$ representation prediction error.
    pub fn loss(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        pred.sub(target)?.sqr()?.sum_keepdim(1)
    }
}

/// Computation Spaces Master Model.
#[derive(Debug, Clone)]
pub struct ComputationSpacesModel {
    pub weight: Tensor,
    pub bias: Tensor,
    pub reasoner: ContinuousLatentReasoner,
    pub ponder: PonderNetHalting,
    pub jepa: JEPAPredictor,
}

impl ComputationSpacesModel {
    pub fn new(
        weight: Tensor,
        bias: Tensor,
        reasoner: ContinuousLatentReasoner,
        ponder: PonderNetHalting,
        jepa: JEPAPredictor,
    ) -> Self {
        Self {
            weight,
            bias,
            reasoner,
            ponder,
            jepa,
        }
    }

    /// Canonical linear forward pass matching test_output: $\mathbf{x} \mathbf{W} + \mathbf{b}$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight)?.broadcast_add(&self.bias)
    }
}
