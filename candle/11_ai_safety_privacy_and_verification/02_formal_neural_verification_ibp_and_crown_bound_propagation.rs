//! Formal Neural Verification IBP and CROWN Bound Propagation
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Interval Bound Propagation for linear layer: y = x @ W + b.
pub fn ibp_linear(
    l_in: &Tensor,
    u_in: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let l_vec = l_in.to_vec2::<f32>()?;
    let u_vec = u_in.to_vec2::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let b_vec = bias.to_vec1::<f32>()?;

    let b = l_vec.len();
    let d_in = w_vec.len();
    let d_out = w_vec[0].len();

    let mut l_out = vec![vec![0.0f32; d_out]; b];
    let mut u_out = vec![vec![0.0f32; d_out]; b];

    for i in 0..b {
        for k in 0..d_out {
            let mut mu_val = b_vec[k];
            let mut r_val = 0.0f32;
            for j in 0..d_in {
                let mu_in = 0.5 * (u_vec[i][j] + l_vec[i][j]);
                let r_in = 0.5 * (u_vec[i][j] - l_vec[i][j]);
                let w = w_vec[j][k];
                mu_val += mu_in * w;
                r_val += r_in * w.abs();
            }
            l_out[i][k] = mu_val - r_val;
            u_out[i][k] = mu_val + r_val;
        }
    }

    Ok((
        Tensor::new(l_out, l_in.device())?,
        Tensor::new(u_out, u_in.device())?,
    ))
}

/// Interval Bound Propagation for ReLU activation.
pub fn ibp_relu(l_in: &Tensor, u_in: &Tensor) -> Result<(Tensor, Tensor)> {
    let l_vec = l_in.to_vec2::<f32>()?;
    let u_vec = u_in.to_vec2::<f32>()?;
    let b = l_vec.len();
    let d = l_vec[0].len();

    let mut l_out = vec![vec![0.0f32; d]; b];
    let mut u_out = vec![vec![0.0f32; d]; b];

    for i in 0..b {
        for j in 0..d {
            l_out[i][j] = l_vec[i][j].max(0.0);
            u_out[i][j] = u_vec[i][j].max(0.0);
        }
    }

    Ok((
        Tensor::new(l_out, l_in.device())?,
        Tensor::new(u_out, u_in.device())?,
    ))
}

/// CROWN linear relaxation parameters (s_L, b_L, s_U, b_U) for ReLU activation on [l, u].
pub fn crown_relu_relaxation(
    l_in: &Tensor,
    u_in: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let l_vec = l_in.to_vec2::<f32>()?;
    let u_vec = u_in.to_vec2::<f32>()?;
    let b = l_vec.len();
    let d = l_vec[0].len();

    let mut s_l = vec![vec![0.0f32; d]; b];
    let mut b_l = vec![vec![0.0f32; d]; b];
    let mut s_u = vec![vec![0.0f32; d]; b];
    let mut b_u = vec![vec![0.0f32; d]; b];

    for i in 0..b {
        for j in 0..d {
            let l = l_vec[i][j];
            let u = u_vec[i][j];

            if l >= 0.0 {
                // Strictly active
                s_l[i][j] = 1.0;
                b_l[i][j] = 0.0;
                s_u[i][j] = 1.0;
                b_u[i][j] = 0.0;
            } else if u <= 0.0 {
                // Strictly inactive
                s_l[i][j] = 0.0;
                b_l[i][j] = 0.0;
                s_u[i][j] = 0.0;
                b_u[i][j] = 0.0;
            } else {
                // Crossing neuron
                let diff = (u - l).max(1e-8);
                s_u[i][j] = u / diff;
                b_u[i][j] = -(u * l) / diff;

                // Adaptive lower slope: 1.0 if u > |l| else 0.0
                s_l[i][j] = if u > -l { 1.0 } else { 0.0 };
                b_l[i][j] = 0.0;
            }
        }
    }

    Ok((
        Tensor::new(s_l, l_in.device())?,
        Tensor::new(b_l, l_in.device())?,
        Tensor::new(s_u, u_in.device())?,
        Tensor::new(b_u, u_in.device())?,
    ))
}

/// CROWN backward linear bound propagation for 2-layer MLP:
/// f(x) = ReLU(x @ W1 + b1) @ W2 + b2, computing certified bounds on margin f(x) @ c.
pub fn crown_2layer_bounds(
    x_l: &Tensor,
    x_u: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
    c: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let xl_vec = x_l.to_vec2::<f32>()?;
    let xu_vec = x_u.to_vec2::<f32>()?;
    let w1_vec = w1.to_vec2::<f32>()?;
    let b1_vec = b1.to_vec1::<f32>()?;
    let w2_vec = w2.to_vec2::<f32>()?;
    let b2_vec = b2.to_vec1::<f32>()?;
    let c_vec = c.to_vec1::<f32>()?;

    let b = xl_vec.len();
    let d_in = xl_vec[0].len();
    let h = w2_vec.len();
    let k = c_vec.len();

    // 1. Forward IBP for hidden layer
    let (z1_l_t, z1_u_t) = ibp_linear(x_l, x_u, w1, b1)?;
    let (sl_t, bl_t, su_t, bu_t) = crown_relu_relaxation(&z1_l_t, &z1_u_t)?;

    let sl_vec = sl_t.to_vec2::<f32>()?;
    let bl_vec = bl_t.to_vec2::<f32>()?;
    let su_vec = su_t.to_vec2::<f32>()?;
    let bu_vec = bu_t.to_vec2::<f32>()?;

    // Vector v = W2 @ c of length H
    let mut v = vec![0.0f32; h];
    for j in 0..h {
        let mut sum = 0.0f32;
        for m in 0..k {
            sum += w2_vec[j][m] * c_vec[m];
        }
        v[j] = sum;
    }

    // Scalar b2 @ c
    let mut b2_c = 0.0f32;
    for m in 0..k {
        b2_c += b2_vec[m] * c_vec[m];
    }

    let mut crown_lower = vec![0.0f32; b];
    let mut crown_upper = vec![0.0f32; b];

    for i in 0..b {
        // Lower bound linear backward propagation
        let mut lambda_l = vec![0.0f32; h];
        let mut bias_relu_l = 0.0f32;
        // Upper bound linear backward propagation
        let mut lambda_u = vec![0.0f32; h];
        let mut bias_relu_u = 0.0f32;

        for j in 0..h {
            let vj = v[j];
            if vj >= 0.0 {
                lambda_l[j] = sl_vec[i][j] * vj;
                bias_relu_l += bl_vec[i][j] * vj;

                lambda_u[j] = su_vec[i][j] * vj;
                bias_relu_u += bu_vec[i][j] * vj;
            } else {
                lambda_l[j] = su_vec[i][j] * vj;
                bias_relu_l += bu_vec[i][j] * vj;

                lambda_u[j] = sl_vec[i][j] * vj;
                bias_relu_u += bl_vec[i][j] * vj;
            }
        }

        // a_l and a_u = lambda @ W1^T of length D_in
        let mut a_l = vec![0.0f32; d_in];
        let mut a_u = vec![0.0f32; d_in];
        for d in 0..d_in {
            let mut sum_l = 0.0f32;
            let mut sum_u = 0.0f32;
            for j in 0..h {
                sum_l += lambda_l[j] * w1_vec[d][j];
                sum_u += lambda_u[j] * w1_vec[d][j];
            }
            a_l[d] = sum_l;
            a_u[d] = sum_u;
        }

        let mut gamma_l = bias_relu_l + b2_c;
        let mut gamma_u = bias_relu_u + b2_c;
        for j in 0..h {
            gamma_l += lambda_l[j] * b1_vec[j];
            gamma_u += lambda_u[j] * b1_vec[j];
        }

        // Input box optimization
        let mut low_val = gamma_l;
        let mut up_val = gamma_u;
        for d in 0..d_in {
            let mu_x = 0.5 * (xu_vec[i][d] + xl_vec[i][d]);
            let r_x = 0.5 * (xu_vec[i][d] - xl_vec[i][d]);

            low_val += a_l[d] * mu_x - a_l[d].abs() * r_x;
            up_val += a_u[d] * mu_x + a_u[d].abs() * r_x;
        }

        crown_lower[i] = low_val;
        crown_upper[i] = up_val;
    }

    Ok((
        Tensor::new(crown_lower, x_l.device())?,
        Tensor::new(crown_upper, x_u.device())?,
    ))
}
