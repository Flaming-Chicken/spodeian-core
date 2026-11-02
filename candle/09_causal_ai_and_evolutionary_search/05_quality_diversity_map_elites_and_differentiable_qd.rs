//! Quality Diversity Map Elites and Differentiable Qd
//! in candle/09_causal_ai_and_evolutionary_search.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Map continuous behavioral descriptors to discrete grid coordinates and 1D cell index.
pub fn map_elites_grid_indices(
    descriptors: &Tensor,
    bounds_min: &Tensor,
    bounds_max: &Tensor,
    grid_size: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let n = descriptors.dim(0)?;
    let b_min = bounds_min.flatten_all()?.to_vec1::<f32>()?;
    let b_max = bounds_max.flatten_all()?.to_vec1::<f32>()?;
    let g_size = grid_size.flatten_all()?.to_vec1::<f32>()?;
    let desc_vec = descriptors.to_vec2::<f32>()?;

    let mut grid_idx = Vec::with_capacity(n * 2);
    let mut cell_idx = Vec::with_capacity(n);

    for row in desc_vec {
        let mut row_idx = [0.0f32; 2];
        for k in 0..2 {
            let scale = (row[k] - b_min[k]) / (b_max[k] - b_min[k] + 1e-8);
            let idx = (scale * g_size[k]).floor().clamp(0.0, g_size[k] - 1.0);
            row_idx[k] = idx;
        }
        grid_idx.push(row_idx[0]);
        grid_idx.push(row_idx[1]);
        cell_idx.push(row_idx[0] * g_size[1] + row_idx[1]);
    }

    let dev = descriptors.device();
    let grid_t = Tensor::from_vec(grid_idx, (n, 2), dev)?;
    let cell_t = Tensor::from_vec(cell_idx, (n,), dev)?;

    Ok((grid_t, cell_t))
}

/// Update 1D archive fitness with candidate fitness if candidate achieves higher return.
pub fn map_elites_archive_update(
    archive_fitness: &Tensor,
    candidate_fitness: &Tensor,
    cell_indices: &Tensor,
) -> Result<Tensor> {
    let mut arch = archive_fitness.flatten_all()?.to_vec1::<f32>()?;
    let cand = candidate_fitness.flatten_all()?.to_vec1::<f32>()?;
    let cells = cell_indices.flatten_all()?.to_vec1::<f32>()?;

    for i in 0..cand.len() {
        let c = cells[i] as usize;
        if cand[i] > arch[c] {
            arch[c] = cand[i];
        }
    }

    Tensor::from_vec(arch, archive_fitness.shape(), archive_fitness.device())
}

/// Differentiable Quality Diversity (DQD) multi-objective directional step.
pub fn dqd_gradient_arborescence_step(
    grad_f: &Tensor,
    jac_b: &Tensor,
    coeffs: &Tensor,
) -> Result<Tensor> {
    let c_vec = coeffs.flatten_all()?.to_vec1::<f32>()?;
    let mut step = grad_f.affine(c_vec[0] as f64, 0.0)?;
    let k = jac_b.dim(0)?;
    for i in 0..k {
        let row = jac_b.get(i)?;
        let term = row.affine(c_vec[i + 1] as f64, 0.0)?;
        step = step.add(&term)?;
    }
    Ok(step)
}

/// Gram-Schmidt orthogonalization of behavioral descriptor gradients against objective gradient.
pub fn dqd_orthogonalize_behavior_gradients(
    grad_f: &Tensor,
    jac_b: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let norm_sq = (grad_f.mul(grad_f)?.sum_all()?.to_scalar::<f32>()? as f64) + eps;
    let proj_coeffs = jac_b
        .matmul(&grad_f.unsqueeze(1)?)?
        .squeeze(1)?
        .affine(1.0 / norm_sq, 0.0)?;
    let ortho = jac_b.sub(
        &proj_coeffs
            .unsqueeze(1)?
            .broadcast_mul(&grad_f.unsqueeze(0)?)?,
    )?;
    Ok(ortho)
}
