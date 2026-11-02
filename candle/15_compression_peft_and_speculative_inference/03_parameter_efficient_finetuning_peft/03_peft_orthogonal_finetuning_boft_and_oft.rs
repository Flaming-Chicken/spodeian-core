//! PEFT Orthogonal Finetuning BOFT and Oft
//! in candle/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Invert a square 2D matrix using Gauss-Jordan elimination with partial pivoting.
fn invert_matrix_2d(a: &Tensor) -> Result<Tensor> {
    let shape = a.dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        candle_core::bail!("matrix must be square 2D");
    }
    let n = shape[0];
    let a_data = a.to_vec2::<f32>()?;
    let mut mat = vec![vec![0.0f32; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = a_data[i][j];
        }
        mat[i][n + i] = 1.0;
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-12 {
            candle_core::bail!("singular matrix");
        }
        mat.swap(col, pivot_row);
        let pivot = mat[col][col];
        for j in 0..(2 * n) {
            mat[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = mat[row][col];
                for j in 0..(2 * n) {
                    mat[row][j] -= factor * mat[col][j];
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = mat[i][n + j];
        }
    }
    Tensor::from_vec(inv, (n, n), a.device())
}

/// Construct orthogonal matrix R from skew-symmetric matrix Q via Cayley transform:
/// R = (I + Q)^(-1) (I - Q)
pub fn cayley_orthogonal(q_skew: &Tensor) -> Result<Tensor> {
    let shape = q_skew.dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        candle_core::bail!("q_skew must be a 2D square matrix");
    }
    let n = shape[0];
    let eye = Tensor::eye(n, candle_core::DType::F32, q_skew.device())?;
    let ipq = eye.add(q_skew)?;
    let imq = eye.sub(q_skew)?;
    let inv_ipq = invert_matrix_2d(&ipq)?;
    inv_ipq.matmul(&imq)
}

/// Block-Diagonal Orthogonal Finetuning (OFT) forward pass.
pub fn block_oft_forward(x: &Tensor, w0: &Tensor, q_blocks: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = q_blocks.dims();
    if dims.len() != 3 || dims[1] != dims[2] {
        candle_core::bail!("q_blocks must have shape [num_blocks, block_size, block_size]");
    }
    let num_blocks = dims[0];
    let block_size = dims[1];
    let d_in = num_blocks * block_size;

    let mut r_blocks = Vec::with_capacity(num_blocks);
    for k in 0..num_blocks {
        let q_k = q_blocks.get(k)?;
        let q_k_t = q_k.t()?;
        let skew_k = (q_k.sub(&q_k_t)? * 0.5)?;
        let r_k = cayley_orthogonal(&skew_k)?;
        r_blocks.push(r_k.to_vec2::<f32>()?);
    }

    let mut r_full_vec = vec![0.0f32; d_in * d_in];
    for (k, r_k) in r_blocks.iter().enumerate() {
        let offset = k * block_size;
        for i in 0..block_size {
            for j in 0..block_size {
                let row = offset + i;
                let col = offset + j;
                r_full_vec[row * d_in + col] = r_k[i][j];
            }
        }
    }

    let r_full = Tensor::from_vec(r_full_vec, (d_in, d_in), x.device())?;
    let w_oft = r_full.matmul(w0)?;
    let h = x.matmul(&w_oft)?;
    Ok((h, r_full))
}

/// Butterfly Orthogonal Finetuning (BOFT) forward pass.
pub fn butterfly_boft_forward(
    x: &Tensor,
    w0: &Tensor,
    q_blocks_layer1: &Tensor,
    q_blocks_layer2: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (_, r1) = block_oft_forward(x, w0, q_blocks_layer1)?;
    let (_, r2) = block_oft_forward(x, w0, q_blocks_layer2)?;
    let r_total = r2.matmul(&r1)?;
    let w_boft = r_total.matmul(w0)?;
    let h = x.matmul(&w_boft)?;
    Ok((h, r_total))
}
