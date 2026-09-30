use spodeian_ml_utils::quantization::*;

#[test]
fn test_sqnr_and_cosine_similarity() {
    let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let identical = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(compute_sqnr(&identical, &reference) >= 100.0);
    assert!((compute_cosine_similarity(&identical, &reference) - 1.0).abs() < 1e-5);

    let noisy = vec![1.05, 1.95, 3.02, 3.98, 5.01];
    let sqnr = compute_sqnr(&noisy, &reference);
    assert!(sqnr > 30.0); // Very low noise -> high SQNR (>30 dB)
    let cos = compute_cosine_similarity(&noisy, &reference);
    assert!(cos > 0.999);

    let (mae, rmse) = compute_mae_rmse(&noisy, &reference);
    assert!(mae < 0.05);
    assert!(rmse < 0.05);
}

#[test]
fn test_ternary_matmul_simd() {
    // 4 weights per byte: 0x01 = +1, 0x03 = -1, 0x00 = 0
    // Byte: 0x01 | (0x03 << 2) | (0x00 << 4) | (0x01 << 6) = 1 | (3<<2) | 0 | (1<<6) = 1 + 12 + 64 = 77
    let packed = vec![0b01_00_11_01]; // w0 = 1, w1 = -1, w2 = 0, w3 = 1
    let acts = vec![2.0, 3.0, 10.0, 5.0];
    let mut out = vec![0.0];
    ternary_matmul_simd_f32(&packed, &acts, &mut out, 1.0);

    // Expected: 1*2.0 + (-1)*3.0 + 0*10.0 + 1*5.0 = 2 - 3 + 5 = 4.0
    assert!((out[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_int8_matmul_simd() {
    let weights: Vec<i8> = vec![127, -127, 0, 63];
    let acts = vec![1.0, 2.0, 5.0, 4.0];
    let mut out = vec![0.0];
    int8_matmul_simd_f32(&weights, &acts, &mut out, 1.0);

    // Expected: (127*1 - 127*2 + 0*5 + 63*4) / 127.0 = (127 - 254 + 252) / 127 = 125 / 127 ≈ 0.98425
    let expected = 125.0 / 127.0;
    assert!((out[0] - expected).abs() < 1e-4);
}

#[test]
fn test_posit8_decode_and_lut() {
    let lut = PositDecoder::get_posit8_lut();
    assert_eq!(lut.len(), 256);
    // Byte 0 in posit is 0.0
    assert_eq!(lut[0], 0.0);

    // Byte 0x80 is NaR (Not a Real)
    assert!(lut[0x80].is_nan());

    // Non-zero posit decodes to real values
    let val = PositDecoder::decode_posit8(0x40);
    assert!(val.is_finite());
    assert!(val > 0.0);
}

#[test]
fn test_nf4_quantization_and_matmul() {
    assert_eq!(NF4_QUANTILES.len(), 16);
    let code_zero = quantize_to_nf4(0.0);
    assert_eq!(dequantize_nf4(code_zero), 0.0);

    let code_one = quantize_to_nf4(1.0);
    assert_eq!(dequantize_nf4(code_one), 1.0);

    let code_neg_one = quantize_to_nf4(-1.0);
    assert_eq!(dequantize_nf4(code_neg_one), -1.0);

    // Matmul test
    let packed = vec![(code_one & 0x0F) | ((code_zero & 0x0F) << 4)];
    let acts = vec![2.0, 5.0];
    let mut out = vec![0.0];
    nf4_matmul_simd_f32(&packed, &acts, &mut out, 1.0);
    // 1.0 * 2.0 + 0.0 * 5.0 = 2.0
    assert!((out[0] - 2.0).abs() < 1e-4);
}

#[test]
fn test_fp16_bf16_conversion() {
    let val = std::f32::consts::PI;
    let bf16_bits = encode_bf16(val);
    let bf16_val = decode_bf16(bf16_bits);
    assert!((bf16_val - val).abs() < 0.02);

    let fp16_bits = encode_fp16(val);
    let fp16_val = decode_fp16(fp16_bits);
    assert!((fp16_val - val).abs() < 0.005);
}

#[test]
fn test_fp8_decoding() {
    let v_e4m3 = decode_fp8_e4m3(0b0011_1000); // sign 0, exp 7 (bias 7 -> 0), mant 0 -> 1.0 * 2^0 = 1.0
    assert!((v_e4m3 - 1.0).abs() < 1e-5);

    let v_e5m2 = decode_fp8_e5m2(0b0011_1100); // sign 0, exp 15 (bias 15 -> 0), mant 0 -> 1.0 * 2^0 = 1.0
    assert!((v_e5m2 - 1.0).abs() < 1e-5);
}


#[test]
fn test_bnn_matmul() {
    // 1-bit packed: bit 0 = 1 (+1.0), bit 1 = 0 (-1.0)
    let packed = vec![0b00000001]; // bit 0 is 1 (+1), bit 1..7 are 0 (-1)
    let acts = vec![1.0, 1.0];
    let mut out = vec![0.0];
    bnn_matmul_simd_f32(&packed, &acts, &mut out, 1.0);
    // Expected: (+1.0)*1.0 + (-1.0)*1.0 = 0.0
    assert!((out[0] - 0.0).abs() < 1e-5);
}
