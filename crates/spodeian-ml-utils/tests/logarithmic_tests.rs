//! Integration tests for native Logarithmic Number System (LNS) arithmetic.

use spodeian_ml_utils::logarithmic::{log8_dot_product, quantize_to_log8, Log8};

#[test]
fn test_log8_f32_roundtrip_precision() {
    let test_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, -1.0, -2.0, 0.125];

    for &v in &test_values {
        let log_val = Log8::from_f32(v);
        let reconstructed = log_val.to_f32();
        if v == 0.0 {
            assert_eq!(reconstructed, 0.0);
        } else {
            // Log8 with 0.25 exponent step has at worst ~9% relative error across 7 orders of magnitude
            let rel_err = ((reconstructed - v) / v).abs();
            assert!(rel_err < 0.10, "Failed for {v}: reconstructed {reconstructed}");
        }
    }
}

#[test]
fn test_log8_multiplication() {
    let a = Log8::from_f32(2.0);
    let b = Log8::from_f32(4.0);
    let prod = a.mul(b);
    let res = prod.to_f32();

    // 2.0 * 4.0 = 8.0
    assert!((res - 8.0).abs() < 1e-3);

    // Negative numbers
    let c = Log8::from_f32(-3.0);
    let prod_neg = a.mul(c);
    let res_neg = prod_neg.to_f32();
    assert!(res_neg < 0.0);
    assert!((res_neg - (-6.0)).abs() < 0.5);
}

#[test]
fn test_log8_dot_product() {
    let v1 = vec![1.0, 2.0, -1.5, 0.5];
    let v2 = vec![2.0, 0.5, 2.0, 4.0];

    // Expected dot product: 1*2 + 2*0.5 + (-1.5)*2 + 0.5*4 = 2 + 1 - 3 + 2 = 2.0
    let l1 = quantize_to_log8(&v1);
    let l2 = quantize_to_log8(&v2);

    let dot = log8_dot_product(&l1, &l2);
    assert!((dot - 2.0).abs() < 0.35);
}
