use candle_core::{Device, Result, safetensors::load};
use sml_candle::c00_foundations::t02_tensors_views_strides_and_memory_layouts::MemoryLayout;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/02_tensors_views_strides_and_memory_layouts.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight = tensors.get("layers.weight").expect("missing layers.weight");
    let bias = tensors.get("layers.bias").expect("missing layers.bias");
    let test_input = tensors.get("test_input").expect("missing test_input");
    let golden_output = tensors.get("test_output").expect("missing test_output");

    // Forward pass: Y = X @ W + b
    let y_candle = test_input.matmul(weight)?.broadcast_add(bias)?;

    // Direct comparison of golden_output to Candle: ||Y_candle - Y_golden||_inf < 1e-5
    common::assert_max_diff(&y_candle, golden_output, 1e-5)?;

    // Memory layout and stride arithmetic tests
    let shape = vec![4, 2];
    let strides = MemoryLayout::standard_strides(&shape);
    assert_eq!(strides, vec![2, 1]);
    let idx = MemoryLayout::linear_index(&[2, 1], &strides);
    assert_eq!(idx, 5);

    // Sliding window validation on test_input
    let windows = MemoryLayout::sliding_windows_1d(test_input, 2, 1)?;
    assert_eq!(windows.dims(), &[4, 3, 2]);

    Ok(())
}
