//! PEFT Representation Finetuning Loreft
//! in candle/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper for 3D tensor multiplied by 2D weight matrix: [B, S, D] @ [D, Out] -> [B, S, Out].
fn matmul_3d_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() == 3 {
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let x_2d = x.reshape((b * s, d))?;
        let out_2d = x_2d.matmul(w)?;
        let out_d = out_2d.dim(1)?;
        out_2d.reshape((b, s, out_d))
    } else {
        x.matmul(w)
    }
}

/// Compute LoReFT representation delta: delta_h = R^T (W h + b - R h).
pub fn loreft_subspace_delta(
    h: &Tensor,
    r_matrix: &Tensor,
    w_matrix: &Tensor,
    b_vec: &Tensor,
) -> Result<Tensor> {
    let r_t = r_matrix.t()?;
    let w_t = w_matrix.t()?;

    let proj_r = matmul_3d_2d(h, &r_t)?;
    let proj_w = matmul_3d_2d(h, &w_t)?.broadcast_add(b_vec)?;
    let delta_s = proj_w.sub(&proj_r)?;

    matmul_3d_2d(&delta_s, r_matrix)
}

/// LoReFT intervention at selective token positions: h_out = h + mask * delta_h.
pub fn loreft_intervene(
    h: &Tensor,
    r_matrix: &Tensor,
    w_matrix: &Tensor,
    b_vec: &Tensor,
    position_mask: &Tensor,
) -> Result<Tensor> {
    let delta_h = loreft_subspace_delta(h, r_matrix, w_matrix, b_vec)?;
    let mask_3d = position_mask.unsqueeze(2)?;
    let gated_delta = delta_h.broadcast_mul(&mask_3d)?;
    h.add(&gated_delta)
}
