//! Integration test for Category 03 Topic 01: Activation Functions Zoo (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t01_activation_functions_zoo::{
    geglu_forward, leaky_relu_forward, relu_forward, selu_forward, silu_forward, snake_forward,
    swiglu_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/01_activation_functions_zoo.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors
        .get("test_in")
        .or_else(|| tensors.get("x_anchor"))
        .expect("test_in");
    let test_out = tensors
        .get("test_out")
        .or_else(|| tensors.get("linear_anchor_out"))
        .expect("test_out");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

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

    // 2. SwiGLU & GeGLU Parity
    let x_gate = tensors.get("x_gate").expect("x_gate");
    let w_gate = tensors.get("w_gate").expect("w_gate");
    let w_up = tensors.get("w_up").expect("w_up");
    let w_down = tensors.get("w_down").expect("w_down");
    let swiglu_gold = tensors.get("swiglu_out").expect("swiglu_out");
    let geglu_gold = tensors.get("geglu_out").expect("geglu_out");

    let swiglu_calc = swiglu_forward(x_gate, w_gate, w_up, w_down)?;
    let diff_swiglu = swiglu_calc
        .sub(swiglu_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_swiglu < 1e-5,
        "SwiGLU error {:.2e} exceeds 1e-5",
        diff_swiglu
    );

    let geglu_calc = geglu_forward(x_gate, w_gate, w_up, w_down)?;
    let diff_geglu = geglu_calc
        .sub(geglu_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_geglu < 1e-5,
        "GeGLU error {:.2e} exceeds 1e-5",
        diff_geglu
    );

    // 3. Snake Activation Parity
    let x_snake = tensors.get("x_snake").expect("x_snake");
    let snake_a = tensors.get("snake_a").expect("snake_a").to_vec1::<f32>()?[0] as f64;
    let snake_gold = tensors.get("snake_out").expect("snake_out");

    let snake_calc = snake_forward(x_snake, snake_a)?;
    let diff_snake = snake_calc
        .sub(snake_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_snake < 1e-5,
        "Snake error {:.2e} exceeds 1e-5",
        diff_snake
    );

    // 4. SELU Activation Parity
    let x_selu = tensors.get("x_selu").expect("x_selu");
    let selu_scale = tensors
        .get("selu_scale")
        .expect("selu_scale")
        .to_vec1::<f32>()?[0] as f64;
    let selu_alpha = tensors
        .get("selu_alpha")
        .expect("selu_alpha")
        .to_vec1::<f32>()?[0] as f64;
    let selu_gold = tensors.get("selu_out").expect("selu_out");

    let selu_calc = selu_forward(x_selu, selu_scale, selu_alpha)?;
    let diff_selu = selu_calc
        .sub(selu_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_selu < 1e-5,
        "SELU error {:.2e} exceeds 1e-5",
        diff_selu
    );

    // 5. Rectified & Smooth Parity
    let x_dead = tensors.get("x_dead").expect("x_dead");
    let relu_gold = tensors.get("relu_out").expect("relu_out");
    let leaky_gold = tensors.get("leaky_relu_out").expect("leaky_relu_out");
    let silu_gold = tensors.get("silu_out").expect("silu_out");

    let relu_calc = relu_forward(x_dead)?;
    let diff_relu = relu_calc
        .sub(relu_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_relu < 1e-5,
        "ReLU error {:.2e} exceeds 1e-5",
        diff_relu
    );

    let leaky_calc = leaky_relu_forward(x_dead, 0.1)?;
    let diff_leaky = leaky_calc
        .sub(leaky_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_leaky < 1e-5,
        "LeakyReLU error {:.2e} exceeds 1e-5",
        diff_leaky
    );

    let silu_calc = silu_forward(x_dead)?;
    let diff_silu = silu_calc
        .sub(silu_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_silu < 1e-5,
        "SiLU error {:.2e} exceeds 1e-5",
        diff_silu
    );

    Ok(())
}
