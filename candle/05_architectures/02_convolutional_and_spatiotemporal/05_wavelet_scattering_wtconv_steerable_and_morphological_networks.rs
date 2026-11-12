//! Wavelet Scattering, WTConv, Steerable & Morphological Networks
//! in candle/05_architectures/02_convolutional_and_spatiotemporal.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// 2D Orthonormal Haar Discrete Wavelet Transform.
pub fn haar_dwt2d(x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (b, c, h, w) = x.dims4()?;
    let h2 = h / 2;
    let w2 = w / 2;

    let reshaped = x.reshape((b, c, h2, 2, w2, 2))?;
    let x00 = reshaped
        .narrow(3, 0, 1)?
        .squeeze(3)?
        .narrow(4, 0, 1)?
        .squeeze(4)?
        .contiguous()?;
    let x01 = reshaped
        .narrow(3, 0, 1)?
        .squeeze(3)?
        .narrow(4, 1, 1)?
        .squeeze(4)?
        .contiguous()?;
    let x10 = reshaped
        .narrow(3, 1, 1)?
        .squeeze(3)?
        .narrow(4, 0, 1)?
        .squeeze(4)?
        .contiguous()?;
    let x11 = reshaped
        .narrow(3, 1, 1)?
        .squeeze(3)?
        .narrow(4, 1, 1)?
        .squeeze(4)?
        .contiguous()?;

    let ll = ((((x00.add(&x01)?).add(&x10)?).add(&x11)?) * 0.5)?;
    let lh = ((((x00.add(&x01)?).sub(&x10)?).sub(&x11)?) * 0.5)?;
    let hl = ((((x00.sub(&x01)?).add(&x10)?).sub(&x11)?) * 0.5)?;
    let hh = ((((x00.sub(&x01)?).sub(&x10)?).add(&x11)?) * 0.5)?;

    Ok((ll, lh, hl, hh))
}

/// 2D Orthonormal Haar Inverse Discrete Wavelet Transform.
pub fn haar_idwt2d(ll: &Tensor, lh: &Tensor, hl: &Tensor, hh: &Tensor) -> Result<Tensor> {
    let (b, c, h2, w2) = ll.dims4()?;
    let h = h2 * 2;
    let w = w2 * 2;

    let y00 = ((((ll.add(lh)?).add(hl)?).add(hh)?) * 0.5)?;
    let y01 = ((((ll.add(lh)?).sub(hl)?).sub(hh)?) * 0.5)?;
    let y10 = ((((ll.sub(lh)?).add(hl)?).sub(hh)?) * 0.5)?;
    let y11 = ((((ll.sub(lh)?).sub(hl)?).add(hh)?) * 0.5)?;

    let row0 = Tensor::stack(&[&y00, &y01], 4)?;
    let row1 = Tensor::stack(&[&y10, &y11], 4)?;

    let grid = Tensor::stack(&[&row0, &row1], 3)?;
    grid.contiguous()?.reshape((b, c, h, w))
}

/// Wavelet Convolution Block forward pass.
pub fn wtconv_forward(
    x: &Tensor,
    wt_ll_w: &Tensor,
    wt_ll_b: &Tensor,
    wt_proj_w: &Tensor,
    wt_proj_b: &Tensor,
) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let (ll, lh, hl, hh) = haar_dwt2d(x)?;

    // Depthwise conv on LL: wt_ll_w has shape [C, 1, 1, 1]
    let ll_w_bcast = wt_ll_w.reshape((1, c, 1, 1))?;
    let ll_b_bcast = wt_ll_b.reshape((1, c, 1, 1))?;
    let ll_proc = ll.broadcast_mul(&ll_w_bcast)?.broadcast_add(&ll_b_bcast)?;

    let x_rec = haar_idwt2d(&ll_proc, &lh, &hl, &hh)?;

    // 1x1 conv projection on x: wt_proj_w is [c_out, c_in, 1, 1]
    let c_out = wt_proj_w.dim(0)?;
    let proj_2d = wt_proj_w.reshape((c_out, c))?;
    let x_flat = x
        .transpose(1, 2)?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((b * h * w, c))?;
    let base_flat = x_flat.matmul(&proj_2d.t()?)?;
    let base = base_flat
        .reshape((b, h, w, c_out))?
        .transpose(2, 3)?
        .transpose(1, 2)?
        .contiguous()?
        .broadcast_add(&wt_proj_b.reshape((1, c_out, 1, 1))?)?;

    base.add(&x_rec)
}

/// Deep Morphological Dilation Layer forward pass with replicate padding and 3x3 structuring element.
pub fn morphological_dilation_forward(x: &Tensor, se: &Tensor) -> Result<Tensor> {
    let (_b, c, h, w) = x.dims4()?;

    // Replicate padding: pad 1 along H and W
    let top = x.narrow(2, 0, 1)?;
    let bottom = x.narrow(2, h - 1, 1)?;
    let padded_h = Tensor::cat(&[&top, x, &bottom], 2)?;

    let left = padded_h.narrow(3, 0, 1)?;
    let right = padded_h.narrow(3, w - 1, 1)?;
    let padded = Tensor::cat(&[&left, &padded_h, &right], 3)?;

    let mut out: Option<Tensor> = None;

    for kh in 0..3 {
        for kw in 0..3 {
            let slice = padded.narrow(2, kh, h)?.narrow(3, kw, w)?.contiguous()?;
            let se_elem = se
                .narrow(1, kh, 1)?
                .narrow(2, kw, 1)?
                .reshape((1, c, 1, 1))?;
            let val = slice.broadcast_add(&se_elem)?;
            out = match out {
                Some(prev) => Some(prev.maximum(&val)?),
                None => Some(val),
            };
        }
    }

    out.ok_or_else(|| candle_core::Error::Msg("Empty kernel in morphological dilation".into()))
}
