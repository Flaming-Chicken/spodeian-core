//! Candle Implementation for Category 04 Topic 21 (Subfolder 03 Topic 07):
//! Scaling Laws, Data Deduplication (MinHash), DoReMi, and Model Collapse.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Chinchilla parametric scaling laws and compute-optimal parameter/data allocation (Hoffmann et al., 2022).
//! 2. DoReMi domain weight optimization via excess loss exponentiated descent (Xie et al., 2023).
//! 3. MinHash Jaccard similarity estimation for scalable deduplication (Broder, 1997).
//! 4. Model collapse recursive variance tracking and real-data accumulation cure (Shumailov et al., 2024).
//! 5. Linear anchor evaluation for transitive multi-track parity.

use candle_core::{Device, Result, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Linear Anchor module for cross-track parity verification.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xw = x.matmul(&self.weight)?;
        xw.broadcast_add(&self.bias)
    }
}

/// Chinchilla parametric loss and compute-optimal allocation (Hoffmann et al., 2022).
///
/// loss = E + A * N^(-alpha) + B * D^(-beta)
/// C = 6 * N * D
/// N* = G_N * C^(beta / (alpha + beta))
/// D* = G_D * C^(alpha / (alpha + beta)), where G_D = 1 / (6 * G_N)
pub fn compute_chinchilla_loss_and_opt(
    n: &Tensor,
    d: &Tensor,
    e_const: f64,
    a_const: f64,
    b_const: f64,
    alpha: f64,
    beta: f64,
    g_n: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dev = n.device();
    let n_val = n.flatten_all()?.to_vec1::<f32>()?[0] as f64;
    let d_val = d.flatten_all()?.to_vec1::<f32>()?[0] as f64;

    let loss_val = e_const + a_const * n_val.powf(-alpha) + b_const * d_val.powf(-beta);

    let exp_a = beta / (alpha + beta);
    let exp_b = alpha / (alpha + beta);
    let budget_c = 6.0 * n_val * d_val;
    let g_d = 1.0 / (6.0 * g_n);

    let n_opt_val = g_n * budget_c.powf(exp_a);
    let d_opt_val = g_d * budget_c.powf(exp_b);

    let loss = Tensor::new(&[loss_val as f32], dev)?;
    let n_opt = Tensor::new(&[n_opt_val as f32], dev)?;
    let d_opt = Tensor::new(&[d_opt_val as f32], dev)?;

    Ok((loss, n_opt, d_opt))
}

/// DoReMi domain weight exponentiated gradient update (Xie et al., 2023).
///
/// excess_loss = losses_model - losses_ref
/// w_next = softmax(log(w_init) + eta * excess_loss)
pub fn doremi_update_step(
    w_init: &Tensor,
    losses_model: &Tensor,
    losses_ref: &Tensor,
    eta: f64,
) -> Result<(Tensor, Tensor)> {
    let excess_loss = losses_model.sub(losses_ref)?;
    let dev = w_init.device();
    let eta_t = Tensor::new(eta as f32, dev)?;

    let log_w = w_init.log()?;
    let weighted_excess = excess_loss.broadcast_mul(&eta_t)?;
    let logits = log_w.add(&weighted_excess)?;

    let w_next = candle_nn::ops::softmax(&logits, 0)?;
    Ok((excess_loss, w_next))
}

/// MinHash pairwise Jaccard similarity estimation (Broder, 1997).
///
/// signatures: [B, K]
/// jaccard_matrix[i, j] = 1/K sum_k (sig[i, k] == sig[j, k])
pub fn minhash_pairwise_jaccard(signatures: &Tensor) -> Result<Tensor> {
    let dev = signatures.device();
    let (b_docs, k_hashes) = signatures.dims2()?;
    let sig_vec = signatures.to_vec2::<f32>()?;

    let mut jaccard_data = vec![0.0f32; b_docs * b_docs];
    let k_f = k_hashes as f32;

    for i in 0..b_docs {
        for j in 0..b_docs {
            let mut matches = 0.0f32;
            for k in 0..k_hashes {
                if (sig_vec[i][k] - sig_vec[j][k]).abs() < 1e-6 {
                    matches += 1.0f32;
                }
            }
            jaccard_data[i * b_docs + j] = matches / k_f;
        }
    }

    Tensor::from_vec(jaccard_data, (b_docs, b_docs), dev)
}

/// Model collapse variance decay and real-data accumulation cure (Shumailov et al., 2024).
///
/// cured_var = (1 - lambda_real) * var_synth + lambda_real * var_real
pub fn model_collapse_variance_cure(
    var_synth: &Tensor,
    var_real: &Tensor,
    lambda_real: f64,
) -> Result<Tensor> {
    let dev = var_synth.device();
    let lam_synth = Tensor::new((1.0 - lambda_real) as f32, dev)?;
    let lam_real = Tensor::new(lambda_real as f32, dev)?;

    let term1 = var_synth.broadcast_mul(&lam_synth)?;
    let term2 = var_real.broadcast_mul(&lam_real)?;
    term1.add(&term2)
}

/// Unified Candle container for Category 04 Topic 21 verification.
pub struct ScalingLawsUnifiedModule {
    pub n_val: Tensor,
    pub d_val: Tensor,
    pub golden_chinchilla_loss: Tensor,
    pub golden_n_opt: Tensor,
    pub golden_d_opt: Tensor,

    pub w_doremi_init: Tensor,
    pub losses_model: Tensor,
    pub losses_ref: Tensor,
    pub golden_excess_loss: Tensor,
    pub golden_w_doremi_next: Tensor,

    pub signatures: Tensor,
    pub golden_jaccard_matrix: Tensor,

    pub var_synth: Tensor,
    pub var_real: Tensor,
    pub golden_cured_var: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl ScalingLawsUnifiedModule {
    pub fn new(weights_path: &Path) -> Result<Self> {
        let dev = &Device::Cpu;
        let tensors = candle_core::safetensors::load(weights_path, dev)?;

        let n_val = tensors.get("n_val").unwrap().clone();
        let d_val = tensors.get("d_val").unwrap().clone();
        let golden_chinchilla_loss = tensors.get("golden_chinchilla_loss").unwrap().clone();
        let golden_n_opt = tensors.get("golden_n_opt").unwrap().clone();
        let golden_d_opt = tensors.get("golden_d_opt").unwrap().clone();

        let w_doremi_init = tensors.get("w_doremi_init").unwrap().clone();
        let losses_model = tensors.get("losses_model").unwrap().clone();
        let losses_ref = tensors.get("losses_ref").unwrap().clone();
        let golden_excess_loss = tensors.get("golden_excess_loss").unwrap().clone();
        let golden_w_doremi_next = tensors.get("golden_w_doremi_next").unwrap().clone();

        let signatures = tensors.get("signatures").unwrap().clone();
        let golden_jaccard_matrix = tensors.get("golden_jaccard_matrix").unwrap().clone();

        let var_synth = tensors.get("var_synth").unwrap().clone();
        let var_real = tensors.get("var_real").unwrap().clone();
        let golden_cured_var = tensors.get("golden_cured_var").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            n_val,
            d_val,
            golden_chinchilla_loss,
            golden_n_opt,
            golden_d_opt,
            w_doremi_init,
            losses_model,
            losses_ref,
            golden_excess_loss,
            golden_w_doremi_next,
            signatures,
            golden_jaccard_matrix,
            var_synth,
            var_real,
            golden_cured_var,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. Chinchilla invariant
        let (loss, n_opt, d_opt) = compute_chinchilla_loss_and_opt(
            &self.n_val,
            &self.d_val,
            1.69,
            40.0,
            50.0,
            0.34,
            0.28,
            0.6,
        )?;
        let err_l = (&loss - &self.golden_chinchilla_loss)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_n = (&n_opt - &self.golden_n_opt)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_d = (&d_opt - &self.golden_d_opt)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_chinchilla = err_l.max(err_n).max(err_d);
        assert!(
            err_chinchilla < atol,
            "Chinchilla invariant failed: {err_chinchilla:.2e} >= {atol:.2e}"
        );
        errs.insert("chinchilla_err".to_string(), err_chinchilla);

        // 2. DoReMi invariant
        let (excess_l, w_next) = doremi_update_step(
            &self.w_doremi_init,
            &self.losses_model,
            &self.losses_ref,
            1.0,
        )?;
        let err_ex = (&excess_l - &self.golden_excess_loss)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_wn = (&w_next - &self.golden_w_doremi_next)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_doremi = err_ex.max(err_wn);
        assert!(
            err_doremi < atol,
            "DoReMi invariant failed: {err_doremi:.2e} >= {atol:.2e}"
        );
        errs.insert("doremi_err".to_string(), err_doremi);

        // 3. MinHash invariant
        let jaccard_m = minhash_pairwise_jaccard(&self.signatures)?;
        let err_minhash = (&jaccard_m - &self.golden_jaccard_matrix)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_minhash < atol,
            "MinHash invariant failed: {err_minhash:.2e} >= {atol:.2e}"
        );
        errs.insert("minhash_err".to_string(), err_minhash);

        // 4. Model collapse invariant
        let cured_v = model_collapse_variance_cure(&self.var_synth, &self.var_real, 0.25)?;
        let err_collapse = (&cured_v - &self.golden_cured_var)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_collapse < atol,
            "Model Collapse invariant failed: {err_collapse:.2e} >= {atol:.2e}"
        );
        errs.insert("collapse_err".to_string(), err_collapse);

        // 5. Linear Anchor invariant
        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let err_anchor = (&anchor_out - &self.golden_anchor_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_anchor < atol,
            "Anchor invariant failed: {err_anchor:.2e} >= {atol:.2e}"
        );
        errs.insert("anchor_err".to_string(), err_anchor);

        Ok(errs)
    }
}
