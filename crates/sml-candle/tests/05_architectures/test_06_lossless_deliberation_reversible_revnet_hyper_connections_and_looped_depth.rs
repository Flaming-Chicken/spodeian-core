use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth as rev;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth_parity()
-> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();

    // 1. Reversible RevNet Verification
    let f_w1 = weights.get("f_w1").unwrap();
    let f_b1 = weights.get("f_b1").unwrap();
    let f_w2 = weights.get("f_w2").unwrap();
    let f_b2 = weights.get("f_b2").unwrap();
    let g_w1 = weights.get("g_w1").unwrap();
    let g_b1 = weights.get("g_b1").unwrap();
    let g_w2 = weights.get("g_w2").unwrap();
    let g_b2 = weights.get("g_b2").unwrap();

    let expected_rev = weights.get("revnet_output").unwrap();
    let out_rev = rev::revnet_forward(x, f_w1, f_b1, f_w2, f_b2, g_w1, g_b1, g_w2, g_b2)?;
    let diff_rev = out_rev
        .sub(expected_rev)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_rev < 1e-5, "RevNet forward error: {}", diff_rev);

    // Exact Inversion Check
    let x_rec = rev::revnet_inverse(&out_rev, f_w1, f_b1, f_w2, f_b2, g_w1, g_b1, g_w2, g_b2)?;
    let diff_rec = x_rec
        .sub(x)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_rec < 1e-5, "RevNet reconstruction error: {}", diff_rec);

    // 2. Hyper-Connections Verification
    let alpha = weights.get("alpha").unwrap();
    let beta = weights.get("beta").unwrap();
    let hc_w1 = weights.get("hc_w1").unwrap();
    let hc_b1 = weights.get("hc_b1").unwrap();
    let hc_w2 = weights.get("hc_w2").unwrap();
    let hc_b2 = weights.get("hc_b2").unwrap();

    let h1 = x.clone();
    let h2 = (x * 0.5)?;
    let (_, _, out_hc) =
        rev::hyper_connections_forward(&h1, &h2, alpha, beta, hc_w1, hc_b1, hc_w2, hc_b2)?;
    let expected_hc = weights.get("hc_output").unwrap();
    let diff_hc = out_hc
        .sub(expected_hc)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_hc < 1e-5, "Hyper-Connections error: {}", diff_hc);

    // 3. Looped Recurrent Depth Verification
    let loop_w1 = weights.get("loop_w1").unwrap();
    let loop_b1 = weights.get("loop_b1").unwrap();
    let loop_w2 = weights.get("loop_w2").unwrap();
    let loop_b2 = weights.get("loop_b2").unwrap();
    let gamma = weights.get("gamma").unwrap().to_scalar::<f32>()? as f64;

    let out_loop = rev::looped_depth_forward(x, loop_w1, loop_b1, loop_w2, loop_b2, 4, gamma)?;
    let expected_loop = weights.get("looped_output").unwrap();
    let diff_loop = out_loop
        .sub(expected_loop)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_loop < 1e-5, "Looped Depth error: {}", diff_loop);

    Ok(())
}
