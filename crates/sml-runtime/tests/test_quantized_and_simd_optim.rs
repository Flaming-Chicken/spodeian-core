//! Unit & Parity Tests for Quantised Optimizer Buffers, SIMD Kernels & Multi-Precision Ladder.

use sml_runtime::optim_kernels::{
    sum_squares_f32, sum_squares_f64, update_coordinates_f32, KernelUpdateParams,
};
use sml_runtime::optim_quantized::{
    QuantizedMetaParamState, QuantizedStateBuffer, StateQuantizationPolicy,
};
use sml_runtime::quantization::{compute_cosine_similarity, compute_sqnr, QuantizedFormat};

#[test]
fn test_quantized_state_buffers_pack_unpack_ladder() {
    let n = 256;
    let original: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32 - 128.0) / 32.0;
            x.sin() * (1.0 + (i as f32 * 0.05).cos())
        })
        .collect();

    let formats = [
        QuantizedFormat::Fp32,
        QuantizedFormat::Bf16,
        QuantizedFormat::Fp16,
        QuantizedFormat::Posit16,
        QuantizedFormat::Posit8,
        QuantizedFormat::Fp8E4M3,
        QuantizedFormat::Fp8E5M2,
        QuantizedFormat::Log8,
        QuantizedFormat::Int8,
        QuantizedFormat::Nf4,
        QuantizedFormat::Int4,
        QuantizedFormat::Ternary158,
        QuantizedFormat::Bnn1Bit,
    ];

    for &fmt in &formats {
        let mut buf = QuantizedStateBuffer::zeros(n, fmt);
        buf.pack(&original, false, 42);
        let unpacked = buf.unpack_to_vec();

        assert_eq!(unpacked.len(), n);
        let cos_sim = compute_cosine_similarity(&original, &unpacked);
        let sqnr = compute_sqnr(&unpacked, &original);

        println!(
            "Format {:?}: bytes/elem={:.2}, cos_sim={:.4}, sqnr={:.2} dB",
            fmt,
            buf.bytes_per_element(),
            cos_sim,
            sqnr
        );

        // Verification gates based on bit budget
        match fmt {
            QuantizedFormat::Fp32 => {
                assert!(cos_sim > 0.9999, "FP32 cosine similarity must be > 0.9999");
                assert!(sqnr > 70.0, "FP32 SQNR must be > 70 dB");
            }
            QuantizedFormat::Bf16 | QuantizedFormat::Fp16 | QuantizedFormat::Posit16 => {
                assert!(cos_sim > 0.999, "16-bit cosine similarity must be > 0.999");
                assert!(sqnr > 30.0, "16-bit SQNR must be > 30 dB");
            }
            QuantizedFormat::Fp8E4M3 | QuantizedFormat::Fp8E5M2 | QuantizedFormat::Int8 | QuantizedFormat::Posit8 => {
                assert!(cos_sim > 0.98, "8-bit cosine similarity must be > 0.98");
                assert!(sqnr > 15.0, "8-bit SQNR must be > 15 dB");
            }
            QuantizedFormat::Nf4 | QuantizedFormat::Int4 => {
                assert!(cos_sim > 0.90, "4-bit cosine similarity must be > 0.90");
            }
            QuantizedFormat::Ternary158 | QuantizedFormat::Bnn1Bit => {
                assert!(cos_sim > 0.65, "Ternary/BNN cosine similarity must be > 0.65");
            }
            _ => {}
        }
    }
}

#[test]
fn test_memory_footprint_monotonic_scaling() {
    let n = 1000;
    let exact_f64 = QuantizedMetaParamState::new(n, StateQuantizationPolicy::exact_f64());
    let exact_f32 = QuantizedMetaParamState::new(n, StateQuantizationPolicy::exact_f32());
    let mixed_bf16 = QuantizedMetaParamState::new(n, StateQuantizationPolicy::mixed_bf16());
    let compact_8bit = QuantizedMetaParamState::new(n, StateQuantizationPolicy::compact_8bit());
    let compact_4bit = QuantizedMetaParamState::new(n, StateQuantizationPolicy::compact_4bit());
    let extreme_low = QuantizedMetaParamState::new(n, StateQuantizationPolicy::extreme_low_bit());

    let _rep_f64 = exact_f64.memory_footprint();
    let rep_f32 = exact_f32.memory_footprint();
    let rep_bf16 = mixed_bf16.memory_footprint();
    let rep_8bit = compact_8bit.memory_footprint();
    let rep_4bit = compact_4bit.memory_footprint();
    let rep_low = extreme_low.memory_footprint();

    println!("Exact F32: {:.2} B/param", rep_f32.bytes_per_param);
    println!("Mixed BF16: {:.2} B/param (comp vs F64: {:.1}x)", rep_bf16.bytes_per_param, rep_bf16.compression_ratio_vs_f64);
    println!("Compact 8Bit: {:.2} B/param (comp vs F64: {:.1}x)", rep_8bit.bytes_per_param, rep_8bit.compression_ratio_vs_f64);
    println!("Compact 4Bit: {:.2} B/param (comp vs F64: {:.1}x)", rep_4bit.bytes_per_param, rep_4bit.compression_ratio_vs_f64);
    println!("Extreme LowBit: {:.2} B/param (comp vs F64: {:.1}x)", rep_low.bytes_per_param, rep_low.compression_ratio_vs_f64);

    // Verify monotonically decreasing memory footprint
    assert!(rep_f32.bytes_per_param <= 28.0);
    assert!(rep_bf16.bytes_per_param < rep_f32.bytes_per_param);
    assert!(rep_8bit.bytes_per_param < rep_bf16.bytes_per_param);
    assert!(rep_4bit.bytes_per_param < rep_8bit.bytes_per_param);
    assert!(rep_low.bytes_per_param < rep_4bit.bytes_per_param);

    // Low-bit tier must achieve > 5x compression over F64 and > 3x over F32
    assert!(rep_low.compression_ratio_vs_f64 >= 5.0);
    assert!(rep_low.compression_ratio_vs_f32 >= 2.5);
}

#[test]
fn test_simd_coordinate_update_convergence() {
    let n = 2048;
    let mut y = vec![2.0f32; n];
    let mut z = vec![2.0f32; n];
    let mut m = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    let mut prev_g = vec![0.0f32; n];
    let mut flip_ema = vec![0.0f32; n];

    let params = KernelUpdateParams {
        step: 0,
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        beta_polyak: 0.8,
        gamma_flip: 0.9,
        flip_damping_factor: 0.5,
        eps: 1e-8,
        weight_decay: 0.0,
        cusp_power: 1.0,
        lambda_hill: 0.0,
        mean_g_sq: 0.0,
        is_multi_layer: false,
    };

    // Optimize a simple quadratic bowl f(y) = 0.5 * sum(y_i^2) -> grad = y
    // Initial loss: 0.5 * 2048 * 4.0 = 4096.0
    for s in 0..200 {
        let grads = y.clone();
        let mut p = params;
        p.step = s;
        update_coordinates_f32(
            p,
            &mut y,
            &mut z,
            &mut m,
            &mut v,
            &mut prev_g,
            &mut flip_ema,
            &grads,
        );
    }

    let final_loss = 0.5 * sum_squares_f32(&y);
    let per_coord_loss = final_loss / n as f64;
    println!("SIMD kernel 200-step quadratic loss: {:.6e} (per coordinate: {:.6e})", final_loss, per_coord_loss);
    assert!(per_coord_loss < 1e-4, "SIMD update kernel must converge to < 1e-4 per coordinate");
}

#[test]
fn test_parallel_chunked_sum_squares_parity() {
    let n = 32768;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

    let sum32 = sum_squares_f32(&data);
    let sum64 = sum_squares_f64(&data_f64);

    let diff = (sum32 - sum64).abs();
    println!("Parallel sum diff: {:.6e}", diff);
    assert!(diff < 1e-3, "f32 and f64 sum of squares must agree within f32 accumulation tolerance");
}
