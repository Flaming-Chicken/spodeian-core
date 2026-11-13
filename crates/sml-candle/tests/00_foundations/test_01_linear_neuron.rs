use ai_candle::c00_foundations::t01_linear_neuron_and_invertible_transforms::{
    CoordinateTransforms, LinearNeuron,
};
use candle_core::{Device, Result, safetensors::load};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/01_linear_neuron_and_invertible_transforms.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight = tensors.get("layers.weight").expect("missing layers.weight");
    let bias = tensors.get("layers.bias").expect("missing layers.bias");
    let test_input = tensors.get("test_input").expect("missing test_input");
    let golden_output = tensors.get("test_output").expect("missing test_output");

    let neuron = LinearNeuron::new(weight.clone(), bias.clone());
    let y_candle = neuron.forward(test_input)?;

    // Direct comparison of golden_output to Candle: ||Y_candle - Y_golden||_inf < 1e-5
    common::assert_max_diff(&y_candle, golden_output, 1e-5)?;

    // Invertible pass validation
    let x_rec = neuron.inverse_1d(&y_candle)?;
    common::assert_max_diff(&x_rec, test_input, 1e-5)?;

    // Invertible coordinate standardization roundtrip
    let (normed, mean, std) = CoordinateTransforms::z_score(test_input, 1e-7)?;
    let x_rec_z = CoordinateTransforms::undo_z_score(&normed, &mean, &std)?;
    common::assert_max_diff(&x_rec_z, test_input, 1e-5)?;

    Ok(())
}
