use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t11_wavelet_scattering_wtconv_steerable_and_morphological_networks as wt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_wavelet_scattering_wtconv_steerable_and_morphological_networks_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors",
    );
    let weights = load(weights_path, &device)?;

    // 1. Linear anchor verification
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let expected_linear = weights.get("test_output").unwrap();

    let out_linear = wt::linear_anchor(test_in, weight, bias)?;
    let diff_linear = out_linear
        .sub(expected_linear)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error: {}",
        diff_linear
    );

    // 2. Parseval Haar DWT & IDWT Invertibility
    let input = weights.get("input").unwrap();
    let (ll, lh, hl, hh) = wt::haar_dwt2d(input)?;
    let x_rec = wt::haar_idwt2d(&ll, &lh, &hl, &hh)?;
    let diff_dwt = x_rec
        .sub(input)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_dwt < 1e-5,
        "Haar DWT/IDWT reconstruction error: {}",
        diff_dwt
    );

    // 3. WTConv Forward Verification
    let wt_ll_w = weights.get("wt_ll_w").unwrap();
    let wt_ll_b = weights.get("wt_ll_b").unwrap();
    let wt_proj_w = weights.get("wt_proj_w").unwrap();
    let wt_proj_b = weights.get("wt_proj_b").unwrap();
    let expected_wtconv = weights.get("wtconv_output").unwrap();

    let out_wtconv = wt::wtconv_forward(input, wt_ll_w, wt_ll_b, wt_proj_w, wt_proj_b)?;
    let diff_wt = out_wtconv
        .sub(expected_wtconv)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_wt < 1e-5, "WTConv parity error: {}", diff_wt);

    // 4. Morphological Dilation Verification
    let morph_se = weights.get("morph_se").unwrap();
    let expected_morph = weights.get("morph_output").unwrap();

    let out_morph = wt::morphological_dilation_forward(input, morph_se)?;
    let diff_morph = out_morph
        .sub(expected_morph)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_morph < 1e-5,
        "Morphological dilation parity error: {}",
        diff_morph
    );

    Ok(())
}
