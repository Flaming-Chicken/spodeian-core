//! Trellis and E8 Lattice Quantization QuIP# QTIP
//! in candle/14_quantisation/04_vector_lattice_qat_and_lowbit.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Project single 8D vector onto D8 checkerboard lattice using Conway & Sloane algorithm.
pub fn project_d8_slice(x: &[f32]) -> [f32; 8] {
    let mut f = [0.0f32; 8];
    let mut sum_int: i32 = 0;
    for i in 0..8 {
        f[i] = x[i].round();
        sum_int += f[i] as i32;
    }
    if sum_int % 2 != 0 {
        let mut max_err = -1.0f32;
        let mut max_idx = 0;
        for i in 0..8 {
            let err = (x[i] - f[i]).abs();
            if err > max_err {
                max_err = err;
                max_idx = i;
            }
        }
        if x[max_idx] >= f[max_idx] {
            f[max_idx] += 1.0;
        } else {
            f[max_idx] -= 1.0;
        }
    }
    f
}

/// Fast nearest-point projection onto E8 Gosset lattice: E8 = D8 U (D8 + 1/2)
pub fn project_e8_slice(x: &[f32]) -> [f32; 8] {
    let p0 = project_d8_slice(x);
    let mut x_shift = [0.0f32; 8];
    for i in 0..8 {
        x_shift[i] = x[i] - 0.5;
    }
    let mut p1 = project_d8_slice(&x_shift);
    for i in 0..8 {
        p1[i] += 0.5;
    }

    let mut dist0 = 0.0f32;
    let mut dist1 = 0.0f32;
    for i in 0..8 {
        let d0 = x[i] - p0[i];
        let d1 = x[i] - p1[i];
        dist0 += d0 * d0;
        dist1 += d1 * d1;
    }

    if dist0 <= dist1 {
        p0
    } else {
        p1
    }
}

/// Quantize vectors in R^8 onto scaled E8 lattice.
pub fn quantize_e8_lattice(vectors: &Tensor, scale: &Tensor) -> Result<(Tensor, Tensor)> {
    let (n, d) = vectors.dims2()?;
    if d != 8 {
        candle_core::bail!("Vectors must have dimension 8 for E8 lattice, got {}", d);
    }
    let s_val = scale.to_vec2::<f32>()?[0][0];
    let vecs_data = vectors.to_vec2::<f32>()?;

    let mut points_data = Vec::with_capacity(n * 8);
    let mut deq_data = Vec::with_capacity(n * 8);

    for row in vecs_data.iter() {
        let mut scaled_row = [0.0f32; 8];
        for i in 0..8 {
            scaled_row[i] = row[i] / s_val;
        }
        let e8_pt = project_e8_slice(&scaled_row);
        for i in 0..8 {
            points_data.push(e8_pt[i]);
            deq_data.push(e8_pt[i] * s_val);
        }
    }

    let points = Tensor::from_vec(points_data, (n, 8), vectors.device())?;
    let w_deq = Tensor::from_vec(deq_data, (n, 8), vectors.device())?;
    Ok((w_deq, points))
}

/// Trellis-Coded Quantization (TCQ) forward Viterbi decoding over 1D sequence.
pub fn trellis_coded_quantize_1d(
    x: &Tensor,
    subsets: &Tensor,
    transitions: &Tensor,
) -> Result<Tensor> {
    let x_vec = x.to_vec1::<f32>()?;
    let subsets_data = subsets.to_vec2::<f32>()?;
    let trans_data = transitions.to_vec2::<f32>()?;

    let l = x_vec.len();
    let n_states = subsets_data.len();
    let inf = 1e9f32;

    let mut dp = vec![vec![inf; n_states]; l];
    let mut back_state = vec![vec![0usize; n_states]; l];
    let mut back_level = vec![vec![0.0f32; n_states]; l];

    for s in 0..n_states {
        let mut best_cost = inf;
        let mut best_lvl = 0.0f32;
        for &lvl in &subsets_data[s] {
            let cost = (x_vec[0] - lvl).powi(2);
            if cost < best_cost {
                best_cost = cost;
                best_lvl = lvl;
            }
        }
        dp[0][s] = best_cost;
        back_level[0][s] = best_lvl;
    }

    for t in 1..l {
        for s in 0..n_states {
            for bit in 0..2 {
                let next_s = trans_data[s][bit] as usize;
                for &lvl in &subsets_data[next_s] {
                    let cost = dp[t - 1][s] + (x_vec[t] - lvl).powi(2);
                    if cost < dp[t][next_s] {
                        dp[t][next_s] = cost;
                        back_state[t][next_s] = s;
                        back_level[t][next_s] = lvl;
                    }
                }
            }
        }
    }

    let mut best_last = 0;
    let mut min_final = inf;
    for s in 0..n_states {
        if dp[l - 1][s] < min_final {
            min_final = dp[l - 1][s];
            best_last = s;
        }
    }

    let mut recon = vec![0.0f32; l];
    let mut curr_s = best_last;
    for t in (0..l).rev() {
        recon[t] = back_level[t][curr_s];
        curr_s = back_state[t][curr_s];
    }

    Tensor::from_vec(recon, l, x.device())
}

/// Inference forward pass using E8 lattice quantized weight matrix.
pub fn e8_forward(
    x: &Tensor,
    w_orig: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (d_in, d_out) = w_orig.dims2()?;
    let w_vecs = w_orig.reshape(((), 8))?;
    let (w_deq_vecs, _) = quantize_e8_lattice(&w_vecs, scale)?;
    let w_deq = w_deq_vecs.reshape((d_in, d_out))?;
    let mut y = x.matmul(&w_deq)?;
    if let Some(b) = bias {
        y = y.broadcast_add(b)?;
    }
    Ok((y, w_deq))
}
