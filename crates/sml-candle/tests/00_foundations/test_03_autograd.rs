use candle_core::{Device, Result, safetensors::load};
use sml_candle::c00_foundations::t03_autograd_mechanics_complex_wirtinger_and_custom_functions::LinearVjp;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight = tensors.get("layers.weight").expect("missing layers.weight");
    let bias = tensors.get("layers.bias").expect("missing layers.bias");
    let test_input = tensors.get("test_input").expect("missing test_input");
    let golden_output = tensors.get("test_output").expect("missing test_output");

    // Forward pass via LinearVjp
    let y_candle = LinearVjp::forward(test_input, weight, bias)?;

    // Direct comparison of golden_output to Candle: ||Y_candle - Y_golden||_inf < 1e-5
    common::assert_max_diff(&y_candle, golden_output, 1e-5)?;

    // Backward pass VJP verification
    let grad_output = y_candle.sub(golden_output)?;
    let (dx, dw, db) = LinearVjp::backward(&grad_output, test_input, weight)?;
    assert_eq!(dx.dims(), test_input.dims());
    assert_eq!(dw.dims(), weight.dims());
    assert_eq!(db.dims(), bias.dims());

    // Wirtinger complex gradient check: 2 * dL/dz* = (z - y)
    let w_grad = LinearVjp::complex_wirtinger_grad(&y_candle, golden_output)?;
    common::assert_max_diff(&w_grad, &grad_output, 1e-6)?;

    Ok(())
}
