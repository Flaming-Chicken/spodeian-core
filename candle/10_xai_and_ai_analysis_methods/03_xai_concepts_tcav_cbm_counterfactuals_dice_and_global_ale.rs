//! XAI Concepts TCAV CBM Counterfactuals DiCE and Global ALE
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Testing with Concept Activation Vectors (Kim et al., 2018).
pub fn tcav_directional_derivatives(
    activation_grads: &Tensor,
    cav: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cav_vec = cav.to_vec1::<f32>()?;
    let mut cav_norm_sq = 0.0f32;
    for v in &cav_vec {
        cav_norm_sq += v * v;
    }
    let norm = (cav_norm_sq + 1e-8).sqrt();
    let cav_norm_vec: Vec<f32> = cav_vec.iter().map(|&v| v / norm).collect();
    let cav_norm = Tensor::from_vec(cav_norm_vec, (cav_vec.len(), 1), cav.device())?;

    let derivs = activation_grads.matmul(&cav_norm)?.squeeze(1)?;
    let d_vec = derivs.to_vec1::<f32>()?;
    let pos_count = d_vec.iter().filter(|&&x| x > 0.0).count() as f32;
    let score = Tensor::new(pos_count / (d_vec.len() as f32), cav.device())?;

    Ok((derivs, score))
}

/// Concept Bottleneck Model (Koh et al., 2020) test-time expert concept intervention.
pub fn cbm_test_time_intervention(
    pred_concepts: &Tensor,
    true_concepts: &Tensor,
    intervention_mask: &Tensor,
    task_weights: &Tensor,
    task_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let mask_vec = intervention_mask.to_vec1::<f32>()?;
    let one_minus_mask: Vec<f32> = mask_vec.iter().map(|&m| 1.0 - m).collect();
    let omm_t = Tensor::from_vec(
        one_minus_mask,
        (mask_vec.len(),),
        intervention_mask.device(),
    )?;

    let p_scaled = pred_concepts.broadcast_mul(&omm_t)?;
    let t_scaled = true_concepts.broadcast_mul(intervention_mask)?;
    let c_int = (p_scaled + t_scaled)?;
    let y_pred = c_int.matmul(task_weights)?.broadcast_add(task_bias)?;

    Ok((c_int, y_pred))
}

/// DiCE Determinantal Point Process diversity metric (Mothilal et al., 2020).
pub fn dice_dpp_diversity(counterfactuals: &Tensor) -> Result<Tensor> {
    let cfs = counterfactuals.to_vec2::<f32>()?;
    let k = cfs.len();
    let mut kernel = vec![vec![0.0f32; k]; k];

    for i in 0..k {
        for j in 0..k {
            let mut dist_sq = 0.0f32;
            for d in 0..cfs[i].len() {
                let diff = cfs[i][d] - cfs[j][d];
                dist_sq += diff * diff;
            }
            kernel[i][j] = 1.0 / (1.0 + (dist_sq + 1e-8).sqrt());
        }
    }

    let det = if k == 3 {
        let m = &kernel;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    } else {
        1.0f32
    };

    Tensor::new(det, counterfactuals.device())
}

/// Accumulated Local Effects (Apley & Zhu, 2020) on 1D feature grid.
pub fn ale_accumulated_local_effects(interval_diffs: &Tensor) -> Result<Tensor> {
    let diffs = interval_diffs.to_vec1::<f32>()?;
    let mut cumsum = Vec::with_capacity(diffs.len());
    let mut running = 0.0f32;
    for d in &diffs {
        running += d;
        cumsum.push(running);
    }
    let mean = cumsum.iter().sum::<f32>() / (cumsum.len() as f32);
    let centered: Vec<f32> = cumsum.iter().map(|&x| x - mean).collect();

    Tensor::from_vec(centered, (diffs.len(),), interval_diffs.device())
}
