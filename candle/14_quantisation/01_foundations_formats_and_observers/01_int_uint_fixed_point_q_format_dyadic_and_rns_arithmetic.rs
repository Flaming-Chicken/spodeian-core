//! INT UINT Fixed Point Q Format Dyadic and RNS Arithmetic
//! in candle/14_quantisation/01_foundations_formats_and_observers.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Asymmetric Affine INT8/UINT8 Quantization.
pub fn affine_quantize_dequantize(
    x: &Tensor,
    num_bits: usize,
) -> Result<(Tensor, Tensor, f64, f64)> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let qmax = ((1usize << num_bits) - 1) as f32;
    let xmin = vec.iter().fold(f32::INFINITY, |acc, &v| acc.min(v));
    let xmax = vec.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));

    let scale = (1e-8f32).max((xmax - xmin) / qmax);
    let zero_point = (-xmin / scale).round().clamp(0.0, qmax);

    let mut q_vec = Vec::with_capacity(vec.len());
    let mut dequant_vec = Vec::with_capacity(vec.len());

    for &v in &vec {
        let q = (v / scale + zero_point).round().clamp(0.0, qmax);
        let deq = (q - zero_point) * scale;
        q_vec.push(q);
        dequant_vec.push(deq);
    }

    let q_t = Tensor::from_vec(q_vec, x.shape(), x.device())?;
    let deq_t = Tensor::from_vec(dequant_vec, x.shape(), x.device())?;
    Ok((deq_t, q_t, scale as f64, zero_point as f64))
}

/// Symmetric INT8 Quantization (restricted or full range).
pub fn symmetric_quantize_dequantize(
    x: &Tensor,
    num_bits: usize,
    full_range: bool,
) -> Result<(Tensor, Tensor, f64)> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let qmax = ((1usize << (num_bits - 1)) - 1) as f32;
    let qmin = if full_range {
        -((1usize << (num_bits - 1)) as f32)
    } else {
        -qmax
    };

    let amax = vec.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = (1e-8f32).max(amax / qmax);

    let mut q_vec = Vec::with_capacity(vec.len());
    let mut dequant_vec = Vec::with_capacity(vec.len());

    for &v in &vec {
        let q = (v / scale).round().clamp(qmin, qmax);
        let deq = q * scale;
        q_vec.push(q);
        dequant_vec.push(deq);
    }

    let q_t = Tensor::from_vec(q_vec, x.shape(), x.device())?;
    let deq_t = Tensor::from_vec(dequant_vec, x.shape(), x.device())?;
    Ok((deq_t, q_t, scale as f64))
}

/// Fixed-point Qm.n Quantization.
pub fn fixed_point_qmn(x: &Tensor, m: usize, n: usize) -> Result<Tensor> {
    let scale = (1usize << n) as f32;
    let qmax = ((1usize << (m + n - 1)) - 1) as f32;
    let qmin = -((1usize << (m + n - 1)) as f32);

    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let dequant: Vec<f32> = vec
        .into_iter()
        .map(|v| {
            let q = (v * scale).round().clamp(qmin, qmax);
            q / scale
        })
        .collect();

    Tensor::from_vec(dequant, x.shape(), x.device())
}

/// Dyadic Rescaling simulation.
pub fn dyadic_rescale_tensor(acc_int: &Tensor, m0: i64, shift: i64) -> Result<Tensor> {
    let vec = acc_int.flatten_all()?.to_vec1::<f32>()?;
    let final_shift = shift - 31;

    let res: Vec<f32> = vec
        .into_iter()
        .map(|v| {
            let acc_64 = v as i64;
            let scaled = (acc_64 * m0 + (1i64 << 30)) >> 31;
            let out = if final_shift > 0 {
                (scaled + (1i64 << (final_shift - 1))) >> final_shift
            } else {
                scaled << (-final_shift)
            };
            out as f32
        })
        .collect();

    Tensor::from_vec(res, acc_int.shape(), acc_int.device())
}

/// Extended GCD to compute modular inverse.
fn mod_inverse(a: i64, m: i64) -> i64 {
    let mut t = 0i64;
    let mut newt = 1i64;
    let mut r = m;
    let mut newr = a % m;

    while newr != 0 {
        let quotient = r / newr;
        let temp_t = t - quotient * newt;
        t = newt;
        newt = temp_t;

        let temp_r = r - quotient * newr;
        r = newr;
        newr = temp_r;
    }

    if t < 0 {
        t += m;
    }
    t
}

/// Residue Number System (RNS) carry-free modular multiplication with CRT reconstruction.
pub fn rns_multiply_and_crt(a: &Tensor, b: &Tensor, moduli: &[i64]) -> Result<Tensor> {
    let a_vec = a.flatten_all()?.to_vec1::<f32>()?;
    let b_vec = b.flatten_all()?.to_vec1::<f32>()?;

    let m_total: i64 = moduli.iter().product();
    let m_primes: Vec<i64> = moduli.iter().map(|&m| m_total / m).collect();
    let inverses: Vec<i64> = (0..moduli.len())
        .map(|i| mod_inverse(m_primes[i], moduli[i]))
        .collect();

    let mut out = Vec::with_capacity(a_vec.len());

    for (&av, &bv) in a_vec.iter().zip(b_vec.iter()) {
        let a_int = (av as i64) % m_total;
        let b_int = (bv as i64) % m_total;

        let mut reconstructed = 0i64;
        for i in 0..moduli.len() {
            let m = moduli[i];
            let c_i = ((a_int % m) * (b_int % m)) % m;
            let weight = m_primes[i] * inverses[i];
            reconstructed += c_i * weight;
        }

        out.push((reconstructed % m_total) as f32);
    }

    Tensor::from_vec(out, a.shape(), a.device())
}
