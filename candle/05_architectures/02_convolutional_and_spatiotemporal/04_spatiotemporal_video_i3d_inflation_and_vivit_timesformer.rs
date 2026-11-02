//! Spatiotemporal Video: I3D Weight Inflation & TimeSformer Factorized Space-Time Attention
//! in candle/05_architectures/02_convolutional_and_spatiotemporal.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Inflates a 2D convolutional filter into a 3D spatiotemporal filter with 1/k_t energy scaling.
pub fn i3d_inflate_weights(conv2d_w: &Tensor, k_t: usize) -> Result<Tensor> {
    let exp = conv2d_w.unsqueeze(2)?;
    let slices = vec![&exp; k_t];
    let rep = Tensor::cat(&slices, 2)?;
    rep / (k_t as f64)
}

/// 3D Convolution forward pass with padding (1, 1, 1) and stride 1,
/// evaluated as temporal aggregation of 2D convolutions over adjacent video frames.
pub fn conv3d_forward(x_video: &Tensor, i3d_w: &Tensor, i3d_b: &Tensor) -> Result<Tensor> {
    let (b, _, t, h, w) = x_video.dims5()?;
    let (c_out, _, kt, _, _) = i3d_w.dims5()?;
    let dev = x_video.device();

    let mut t_outs = Vec::with_capacity(t);

    for ti in 0..t {
        let mut frame_sum = Tensor::zeros((b, c_out, h, w), candle_core::DType::F32, dev)?;
        for dt in 0..kt {
            let in_ti = (ti as isize) + (dt as isize) - 1;
            if in_ti >= 0 && in_ti < (t as isize) {
                let frame_slice = x_video
                    .narrow(2, in_ti as usize, 1)?
                    .squeeze(2)?
                    .contiguous()?;
                let kernel_slice = i3d_w.narrow(2, dt, 1)?.squeeze(2)?.contiguous()?;
                let c2d = frame_slice.conv2d(&kernel_slice, 1, 1, 1, 1)?;
                frame_sum = (frame_sum + c2d)?;
            }
        }
        let b_exp = i3d_b.reshape((1, c_out, 1, 1))?;
        let frame_out = frame_sum.broadcast_add(&b_exp)?;
        t_outs.push(frame_out.unsqueeze(2)?);
    }

    let t_refs: Vec<&Tensor> = t_outs.iter().collect();
    Tensor::cat(&t_refs, 2)
}

/// TimeSformer Factorized Space-Time Attention forward pass.
pub fn timesformer_attention_forward(
    z: &Tensor,
    spatial_q_w: &Tensor,
    spatial_k_w: &Tensor,
    spatial_v_w: &Tensor,
    spatial_out_w: &Tensor,
    temporal_q_w: &Tensor,
    temporal_k_w: &Tensor,
    temporal_v_w: &Tensor,
    temporal_out_w: &Tensor,
) -> Result<Tensor> {
    let (b, t, s, d) = z.dims4()?;
    let scale = (d as f64).sqrt();

    // 1. Spatial Attention across S tokens
    let z_s_flat = z.reshape((b * t * s, d))?;
    let q_s = z_s_flat.matmul(spatial_q_w)?.reshape((b * t, s, d))?;
    let k_s = z_s_flat.matmul(spatial_k_w)?.reshape((b * t, s, d))?;
    let v_s = z_s_flat.matmul(spatial_v_w)?.reshape((b * t, s, d))?;
    let scores_s = (q_s.matmul(&k_s.transpose(1, 2)?)? / scale)?;
    let attn_s = candle_nn::ops::softmax_last_dim(&scores_s)?;
    let attn_v_s = attn_s.matmul(&v_s)?;
    let out_s = attn_v_s
        .reshape((b * t * s, d))?
        .matmul(spatial_out_w)?
        .reshape((b, t, s, d))?;
    let z_mid = (z + out_s)?;

    // 2. Temporal Attention across T frames
    let z_t = z_mid.transpose(1, 2)?.contiguous()?;
    let z_t_flat = z_t.reshape((b * s * t, d))?;
    let q_t = z_t_flat.matmul(temporal_q_w)?.reshape((b * s, t, d))?;
    let k_t = z_t_flat.matmul(temporal_k_w)?.reshape((b * s, t, d))?;
    let v_t = z_t_flat.matmul(temporal_v_w)?.reshape((b * s, t, d))?;
    let scores_t = (q_t.matmul(&k_t.transpose(1, 2)?)? / scale)?;
    let attn_t = candle_nn::ops::softmax_last_dim(&scores_t)?;
    let attn_v_t = attn_t.matmul(&v_t)?;
    let out_t = attn_v_t
        .reshape((b * s * t, d))?
        .matmul(temporal_out_w)?
        .reshape((b, s, t, d))?
        .transpose(1, 2)?
        .contiguous()?;

    z_mid + out_t
}
