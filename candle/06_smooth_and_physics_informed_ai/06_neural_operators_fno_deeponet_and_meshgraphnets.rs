//! Neural Operators FNO DeepONet and Meshgraphnets
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Abramowitz and Stegun rational erf approximation (formula 7.1.26).
pub fn erf_approx(x: &Tensor) -> Result<Tensor> {
    let sign = x.sign()?;
    let abs_x = x.abs()?;
    let p = 0.3275911f64;
    let a1 = 0.254829592f64;
    let a2 = -0.284496736f64;
    let a3 = 1.421413741f64;
    let a4 = -1.453152027f64;
    let a5 = 1.061405429f64;

    let one = Tensor::ones_like(&abs_x)?;
    let t = (one.clone() + abs_x.affine(p, 0.0)?)?.recip()?;

    let poly = t
        .affine(a5, a4)?
        .mul(&t)?
        .affine(1.0, a3)?
        .mul(&t)?
        .affine(1.0, a2)?
        .mul(&t)?
        .affine(1.0, a1)?
        .mul(&t)?;

    let exp_neg_x2 = abs_x.sqr()?.neg()?.exp()?;
    let erf_abs = (one - exp_neg_x2.mul(&poly)?)?;
    sign.mul(&erf_abs)
}

/// Exact Gaussian Error Linear Unit (GELU) with erf approximation.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    let inv_sqrt2 = 1.0 / (2.0f64).sqrt();
    let x_over_sqrt2 = x.affine(inv_sqrt2, 0.0)?;
    let erf_val = erf_approx(&x_over_sqrt2)?;
    let one_plus_erf = erf_val.affine(1.0, 1.0)?;
    x.mul(&one_plus_erf)?.affine(0.5, 0.0)
}

/// 1D Discrete Fourier spectral convolution with K truncated modes.
pub fn fno_spectral_conv_1d(
    v: &Tensor,
    r_real: &Tensor,
    r_imag: &Tensor,
    s_len: usize,
    k_modes: usize,
) -> Result<Tensor> {
    let b = v.dim(0)?;
    let c = v.dim(1)?;
    let device = v.device();

    // Construct C and S projection matrices [K, S]
    let mut c_data = Vec::with_capacity(k_modes * s_len);
    let mut s_data = Vec::with_capacity(k_modes * s_len);
    let pi2 = 2.0 * std::f64::consts::PI;

    for k in 0..k_modes {
        for n in 0..s_len {
            let angle = pi2 * (k as f64) * (n as f64) / (s_len as f64);
            c_data.push(angle.cos() as f32);
            s_data.push((-angle.sin()) as f32);
        }
    }

    let c_mat = Tensor::from_vec(c_data, (k_modes, s_len), device)?;
    let s_mat = Tensor::from_vec(s_data, (k_modes, s_len), device)?;

    // Reshape v from [B, C, S] to [B*C, S] for robust 2D matmul with [S, K]
    let v_flat = v.reshape((b * c, s_len))?;
    let vr_flat = v_flat.matmul(&c_mat.t()?)?; // [B*C, K]
    let vi_flat = v_flat.matmul(&s_mat.t()?)?; // [B*C, K]

    let v_r = vr_flat.reshape((b, c, k_modes))?;
    let v_i = vi_flat.reshape((b, c, k_modes))?;

    // Mode-wise complex matmul
    let mut u_r_modes = Vec::with_capacity(k_modes);
    let mut u_i_modes = Vec::with_capacity(k_modes);

    for k in 0..k_modes {
        let vr_k = v_r.narrow(2, k, 1)?.squeeze(2)?; // [B, C]
        let vi_k = v_i.narrow(2, k, 1)?.squeeze(2)?; // [B, C]

        let rr_k = r_real.narrow(0, k, 1)?.squeeze(0)?; // [C, C]
        let ri_k = r_imag.narrow(0, k, 1)?.squeeze(0)?; // [C, C]

        // u_r = vr @ R_real^T - vi @ R_imag^T
        // u_i = vr @ R_imag^T + vi @ R_real^T
        let ur_k = (vr_k.matmul(&rr_k.t()?)? - vi_k.matmul(&ri_k.t()?)?)?;
        let ui_k = (vr_k.matmul(&ri_k.t()?)? + vi_k.matmul(&rr_k.t()?)?)?;

        u_r_modes.push(ur_k.unsqueeze(2)?);
        u_i_modes.push(ui_k.unsqueeze(2)?);
    }

    let ur_refs: Vec<&Tensor> = u_r_modes.iter().collect();
    let ui_refs: Vec<&Tensor> = u_i_modes.iter().collect();
    let u_r = Tensor::cat(&ur_refs, 2)?; // [B, C, K]
    let u_i = Tensor::cat(&ui_refs, 2)?; // [B, C, K]

    // Construct inverse synthesis matrices [K, S]
    let mut m_cos_data = Vec::with_capacity(k_modes * s_len);
    let mut m_sin_data = Vec::with_capacity(k_modes * s_len);

    for k in 0..k_modes {
        let w = if k == 0 { 1.0f64 } else { 2.0f64 };
        for n in 0..s_len {
            let angle = pi2 * (k as f64) * (n as f64) / (s_len as f64);
            m_cos_data.push((w * angle.cos()) as f32);
            m_sin_data.push((w * angle.sin()) as f32);
        }
    }

    let m_cos = Tensor::from_vec(m_cos_data, (k_modes, s_len), device)?;
    let m_sin = Tensor::from_vec(m_sin_data, (k_modes, s_len), device)?;

    let ur_flat = u_r.reshape((b * c, k_modes))?;
    let ui_flat = u_i.reshape((b * c, k_modes))?;

    let term1_flat = ur_flat.matmul(&m_cos)?;
    let term2_flat = ui_flat.matmul(&m_sin)?;
    let kv_flat = (term1_flat - term2_flat)?.affine(1.0 / (s_len as f64), 0.0)?;
    let k_v = kv_flat.reshape((b, c, s_len))?;

    Ok(k_v)
}

/// Complete FNO-1D forward pass: Lifting -> SpectralConv + Bypass -> GELU -> Projection.
pub fn fno_forward(
    fno_input: &Tensor,
    w_p: &Tensor,
    b_p: &Tensor,
    r_real: &Tensor,
    r_imag: &Tensor,
    w_bypass: &Tensor,
    w_q: &Tensor,
    b_q: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let s_len = 8;
    let k_modes = 4;

    let b = fno_input.dim(0)?;
    let c = w_p.dim(0)?;

    // Lifting: [B, 1, S] -> [B, S, 1] -> [B*S, 1] @ [1, C] -> [B*S, C] -> [B, S, C] -> [B, C, S]
    let in_flat = fno_input
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * s_len, 1))?;
    let lift_flat = in_flat.matmul(&w_p.t()?)?;
    let lift_pre = lift_flat
        .reshape((b, s_len, c))?
        .transpose(1, 2)?
        .contiguous()?;
    let b_p_3d = if b_p.dims().len() == 2 {
        b_p.unsqueeze(0)?
    } else {
        b_p.clone()
    };
    let v0 = gelu(&lift_pre.broadcast_add(&b_p_3d)?)?;

    let k_v0 = fno_spectral_conv_1d(&v0, r_real, r_imag, s_len, k_modes)?;

    // Channel bypass: [B, C, S] -> [B*S, C] @ [C, C] -> [B*S, C] -> [B, C, S]
    let v0_flat = v0.transpose(1, 2)?.contiguous()?.reshape((b * s_len, c))?;
    let bypass_flat = v0_flat.matmul(&w_bypass.t()?)?;
    let bypass = bypass_flat
        .reshape((b, s_len, c))?
        .transpose(1, 2)?
        .contiguous()?;
    let v1 = gelu(&(k_v0.clone() + bypass)?)?;

    // Projection: [B, C, S] -> [B*S, C] @ [C, 1] -> [B*S, 1] -> [B, 1, S]
    let v1_flat = v1.transpose(1, 2)?.contiguous()?.reshape((b * s_len, c))?;
    let proj_flat = v1_flat.matmul(&w_q.t()?)?;
    let proj_pre = proj_flat
        .reshape((b, s_len, 1))?
        .transpose(1, 2)?
        .contiguous()?;
    let b_q_3d = if b_q.dims().len() == 2 {
        b_q.unsqueeze(0)?
    } else {
        b_q.clone()
    };
    let fno_output = proj_pre.broadcast_add(&b_q_3d)?;

    Ok((v0, k_v0, v1, fno_output))
}

/// DeepONet operator forward pass: Branch(u) . Trunk(y) + b0.
pub fn deeponet_forward(
    u_sensor: &Tensor,
    y_query: &Tensor,
    w_b: &Tensor,
    b_b: &Tensor,
    w_t: &Tensor,
    b_t: &Tensor,
    b_out0: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let branch_pre = u_sensor.matmul(w_b)?.broadcast_add(b_b)?;
    let branch_feat = gelu(&branch_pre)?;

    let trunk_pre = y_query.matmul(w_t)?.broadcast_add(b_t)?;
    let trunk_feat = gelu(&trunk_pre)?;

    let dot = branch_feat.matmul(&trunk_feat.t()?)?;
    let deeponet_output = dot.broadcast_add(b_out0)?;

    Ok((branch_feat, trunk_feat, deeponet_output))
}

/// MeshGraphNets single-step message passing on unstructured mesh.
pub fn meshgraphnet_step(
    v_nodes: &Tensor,
    src: &[usize],
    dst: &[usize],
    e_feat: &Tensor,
    w_e: &Tensor,
    b_e: &Tensor,
    w_v: &Tensor,
    b_v: &Tensor,
) -> Result<Tensor> {
    let n_nodes = v_nodes.dim(0)?;
    let n_edges = src.len();
    let device = v_nodes.device();

    // Gather node states for each edge
    let mut e_in_rows = Vec::with_capacity(n_edges);
    for k in 0..n_edges {
        let v_src = v_nodes.narrow(0, src[k], 1)?; // [1, 4]
        let v_dst = v_nodes.narrow(0, dst[k], 1)?; // [1, 4]
        let e_k = e_feat.narrow(0, k, 1)?; // [1, 2]
        let row = Tensor::cat(&[&v_src, &v_dst, &e_k], 1)?; // [1, 10]
        e_in_rows.push(row);
    }
    let e_in_refs: Vec<&Tensor> = e_in_rows.iter().collect();
    let e_in = Tensor::cat(&e_in_refs, 0)?; // [n_edges, 10]

    let e_up = gelu(&e_in.matmul(w_e)?.broadcast_add(b_e)?)?; // [n_edges, 4]

    // Aggregate to destination nodes
    let mut e_agg_rows = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        let mut node_agg: Option<Tensor> = None;
        for k in 0..n_edges {
            if dst[k] == i {
                let e_k = e_up.narrow(0, k, 1)?;
                node_agg = match node_agg {
                    Some(cur) => Some((cur + e_k)?),
                    None => Some(e_k),
                };
            }
        }
        let agg_i = node_agg.unwrap_or_else(|| Tensor::zeros((1, 4), DType::F32, device).unwrap());
        e_agg_rows.push(agg_i);
    }
    let e_agg_refs: Vec<&Tensor> = e_agg_rows.iter().collect();
    let e_agg = Tensor::cat(&e_agg_refs, 0)?; // [n_nodes, 4]

    let v_in = Tensor::cat(&[v_nodes, &e_agg], 1)?; // [n_nodes, 8]
    let mgn_output = gelu(&v_in.matmul(w_v)?.broadcast_add(b_v)?)?;

    Ok(mgn_output)
}
