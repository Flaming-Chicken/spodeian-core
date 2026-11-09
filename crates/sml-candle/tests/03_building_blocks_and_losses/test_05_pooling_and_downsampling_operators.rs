//! Integration test for Category 03 Topic 05: Pooling and Downsampling Operators (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t05_pooling_and_downsampling_operators::{
    blur_pool_forward, gem_pool_forward, pixel_unshuffle_forward, pma_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/05_pooling_and_downsampling_operators.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let x_gem = tensors.get("x_gem").expect("x_gem");
    let p_gem = tensors.get("p_gem").expect("p_gem").to_vec1::<f32>()?[0] as f64;
    let y_gem_gold = tensors.get("y_gem").expect("y_gem");

    let x_blur = tensors.get("x_blur").expect("x_blur");
    let blur_kernel = tensors.get("blur_kernel").expect("blur_kernel");
    let y_blur_gold = tensors.get("y_blur").expect("y_blur");

    let x_pixel = tensors.get("x_pixel").expect("x_pixel");
    let y_pixel_gold = tensors.get("y_pixel").expect("y_pixel");

    let x_pma = tensors.get("x_pma").expect("x_pma");
    let s_seed = tensors.get("s_seed").expect("s_seed");
    let y_pma_gold = tensors.get("y_pma").expect("y_pma");

    // 1. Linear Anchor Parity
    let anchor = LinearAnchor::new(weight.clone(), bias.clone());
    let linear_pred = anchor.forward(test_in)?;
    let diff_linear = linear_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error {:.2e} exceeds 1e-5",
        diff_linear
    );

    // 2. GeM Pooling Parity
    let gem_pred = gem_pool_forward(x_gem, p_gem, 1e-6)?;
    let diff_gem = gem_pred
        .sub(y_gem_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_gem < 1e-5,
        "GeM parity error {:.2e} exceeds 1e-5",
        diff_gem
    );

    // 3. BlurPool Parity
    let blur_pred = blur_pool_forward(x_blur, blur_kernel, 2)?;
    let diff_blur = blur_pred
        .sub(y_blur_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_blur < 1e-5,
        "BlurPool parity error {:.2e} exceeds 1e-5",
        diff_blur
    );

    // 4. PixelUnshuffle Parity
    let pixel_pred = pixel_unshuffle_forward(x_pixel, 2)?;
    let diff_pixel = pixel_pred
        .sub(y_pixel_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pixel < 1e-5,
        "PixelUnshuffle parity error {:.2e} exceeds 1e-5",
        diff_pixel
    );

    // 5. PMA Attention Pooling Parity
    let pma_pred = pma_forward(x_pma, s_seed)?;
    let diff_pma = pma_pred
        .sub(y_pma_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pma < 1e-5,
        "PMA parity error {:.2e} exceeds 1e-5",
        diff_pma
    );

    Ok(())
}
