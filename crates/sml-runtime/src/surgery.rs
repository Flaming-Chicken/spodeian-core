//! Model Surgery: Representation Abliteration & LoRA Weight Merging (`sml-runtime::surgery`).
//!
//! Grounding in relevant literature:
//! 1. **Refusal Representation Abliteration (Arditi et al., 2024 / FailSpy)**:
//!    Identifies the low-dimensional subspace mediating refusal and safety censorship
//!    in instruction-tuned LLMs and projects it out of weight matrices in closed-form:
//!    $$W_{\text{ablated}} = W \left( I - \hat{\mathbf{v}} \hat{\mathbf{v}}^T \right) = W - (W \hat{\mathbf{v}}) \hat{\mathbf{v}}^T$$
//! 2. **Low-Rank Adaptation Weight Merging (Hu et al., 2021)**:
//!    Folds additive low-rank adapter factors $B \times A$ directly back into the frozen
//!    linear weights $W_0$ for zero-latency inference and unified export:
//!    $$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B A)$$

use candle_core::{DType, Result, Tensor};

/// Model Surgery and Representation Manipulation Toolkit.
#[derive(Debug, Clone, Default)]
pub struct ModelSurgery;

impl ModelSurgery {
    /// Computes the normalized directional steering/refusal vector $\hat{\mathbf{v}} \in \mathbb{R}^d$
    /// from contrastive activation pairs across positive and negative prompt streams:
    /// $$\mathbf{v} = \frac{1}{N} \sum_{i=1}^N \mathbf{a}_i^{(+)} - \frac{1}{M} \sum_{j=1}^M \mathbf{a}_j^{(-)}, \quad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$
    pub fn compute_contrastive_direction(
        positive_activations: &[Tensor],
        negative_activations: &[Tensor],
    ) -> Result<Tensor> {
        assert!(!positive_activations.is_empty(), "Positive activations cannot be empty");
        assert!(!negative_activations.is_empty(), "Negative activations cannot be empty");

        let mut mean_pos = positive_activations[0].to_dtype(DType::F32)?;
        for act in positive_activations.iter().skip(1) {
            mean_pos = (&mean_pos + &act.to_dtype(DType::F32)?)?;
        }
        mean_pos = (mean_pos / (positive_activations.len() as f64))?;

        let mut mean_neg = negative_activations[0].to_dtype(DType::F32)?;
        for act in negative_activations.iter().skip(1) {
            mean_neg = (&mean_neg + &act.to_dtype(DType::F32)?)?;
        }
        mean_neg = (mean_neg / (negative_activations.len() as f64))?;

        // Difference direction
        let diff = (&mean_pos - &mean_neg)?;
        let flat = diff.flatten_all()?;
        let norm_sq = flat.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = norm_sq.sqrt().max(1e-8);

        diff / (norm as f64)
    }

    /// Performs closed-form orthogonal projection surgery on a weight matrix:
    /// $$W' = W - (W \hat{\mathbf{v}}) \hat{\mathbf{v}}^T$$
    /// Guaranteed to zero out projection along $\hat{\mathbf{v}}$ while minimally altering all orthogonal features.
    pub fn abliterate_direction(weight: &Tensor, direction: &Tensor) -> Result<Tensor> {
        let w_f32 = weight.to_dtype(DType::F32)?;
        let v_f32 = direction.to_dtype(DType::F32)?.flatten_all()?;

        let d_in = v_f32.dim(0)?;
        assert_eq!(
            w_f32.dim(1)?,
            d_in,
            "Direction vector dimension must match weight input dimension"
        );

        // v_col has shape [d_in, 1]
        let v_col = v_f32.reshape((d_in, 1))?;
        // W * v has shape [d_out, 1]
        let wv = w_f32.matmul(&v_col)?;
        // v_row has shape [1, d_in]
        let v_row = v_col.t()?;
        // outer product (W v) * v^T has shape [d_out, d_in]
        let proj = wv.matmul(&v_row)?;

        // W - (W v) v^T
        (&w_f32 - &proj)?.to_dtype(weight.dtype())
    }

    /// Folds LoRA low-rank adapter matrices $A$ and $B$ back into base weight matrix $W_0$:
    /// $$W_{\text{merged}} = W_0 + \text{scaling} \cdot (B A)$$
    pub fn merge_lora_weights(
        base_w: &Tensor,
        lora_a: &Tensor,
        lora_b: &Tensor,
        scaling: f32,
    ) -> Result<Tensor> {
        let w0_f32 = base_w.to_dtype(DType::F32)?;
        let a_f32 = lora_a.to_dtype(DType::F32)?;
        let b_f32 = lora_b.to_dtype(DType::F32)?;

        // delta = B * A with shape [d_out, d_in]
        let delta = b_f32.matmul(&a_f32)?;
        let scaled_delta = (delta * (scaling as f64))?;

        (&w0_f32 + &scaled_delta)?.to_dtype(base_w.dtype())
    }

    /// Measures the directional alignment (cosine similarity) of an activation vector with a target direction.
    pub fn directional_alignment(activation: &Tensor, direction: &Tensor) -> Result<f32> {
        let a_flat = activation.to_dtype(DType::F32)?.flatten_all()?;
        let d_flat = direction.to_dtype(DType::F32)?.flatten_all()?;

        let dot = (&a_flat * &d_flat)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = a_flat.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-8);
        let norm_d = d_flat.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-8);

        Ok(dot / (norm_a * norm_d))
    }
}
