//! Topological Data Analysis Persistent Homology and Singular Learning LLC
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Computes topological feature lifespans pers_i = death_i - birth_i (Edelsbrunner & Harer, 2008).
pub fn persistence_diagram_lifespans(birth_death_pairs: &Tensor) -> Result<Tensor> {
    let pairs = birth_death_pairs.to_vec2::<f32>()?;
    let lifespans: Vec<f32> = pairs.iter().map(|p| p[1] - p[0]).collect();
    Tensor::new(lifespans, birth_death_pairs.device())
}

/// Evaluates level-1 persistence landscape lambda_1(t) = max_i relu(min(t - b_i, d_i - t)) (Bubenik, 2015).
pub fn persistence_landscape_level1(
    birth_death_pairs: &Tensor,
    eval_points: &Tensor,
) -> Result<Tensor> {
    let pairs = birth_death_pairs.to_vec2::<f32>()?;
    let t_vals = eval_points.to_vec1::<f32>()?;
    let mut landscape = vec![0.0f32; t_vals.len()];

    for (col, &t) in t_vals.iter().enumerate() {
        let mut max_val = 0.0f32;
        for p in &pairs {
            let b = p[0];
            let d = p[1];
            let val = (t - b).min(d - t).max(0.0);
            if val > max_val {
                max_val = val;
            }
        }
        landscape[col] = max_val;
    }

    Tensor::new(landscape, eval_points.device())
}

/// Singular Learning Theory Local Learning Coefficient (LLC) hat{lambda} via SGLD (Watanabe, 2009; Lau et al., 2023).
pub fn singular_learning_llc_energy(
    sample_losses: &Tensor,
    loss_star: &Tensor,
    n_samples: f32,
    beta_inv_temp: f32,
) -> Result<Tensor> {
    let losses = sample_losses.to_vec1::<f32>()?;
    let l_star = loss_star.to_scalar::<f32>()?;
    let mean_diff = losses.iter().map(|&l| l - l_star).sum::<f32>() / (losses.len() as f32);
    let llc = (n_samples * beta_inv_temp / 2.0) * mean_diff;
    Tensor::new(llc, sample_losses.device())
}

/// Counts connected components (Betti-0 number beta_0) along a Vietoris-Rips filtration.
pub fn betti_0_filtration_counts(
    pairwise_distances: &Tensor,
    epsilon_thresholds: &Tensor,
) -> Result<Tensor> {
    let dist = pairwise_distances.to_vec2::<f32>()?;
    let epsilons = epsilon_thresholds.to_vec1::<f32>()?;
    let n = dist.len();
    let mut counts = vec![0.0f32; epsilons.len()];

    for (e_idx, &eps) in epsilons.iter().enumerate() {
        let mut visited = vec![false; n];
        let mut comp = 0.0f32;
        for i in 0..n {
            if !visited[i] {
                comp += 1.0;
                let mut queue = vec![i];
                visited[i] = true;
                while !queue.is_empty() {
                    let curr = queue.remove(0);
                    for nbr in 0..n {
                        if dist[curr][nbr] <= eps && !visited[nbr] {
                            visited[nbr] = true;
                            queue.push(nbr);
                        }
                    }
                }
            }
        }
        counts[e_idx] = comp;
    }

    Tensor::new(counts, pairwise_distances.device())
}
