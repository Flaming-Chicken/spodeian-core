//! Integration test for Topic 06: Module Internals, Parametrizations & Meta Device (Candle Track).

use ai_candle::c00_foundations::t06_nn_module_internals_parametrizations_and_meta_device::{
    cayley_transform, MetaTensor, OrthogonalLinear,
};
use candle_core::{safetensors::load, DType, Device, Result, Tensor};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/06_nn_module_internals_parametrizations_and_meta_device.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight_gold = tensors.get("layers.weight").expect("layers.weight");
    let bias_gold = tensors.get("layers.bias").expect("layers.bias");
    let skew_param = tensors.get("layers.skew_param").expect("layers.skew_param");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");

    // 1. Cayley Orthogonal Transform Parity
    let w_calc = cayley_transform(skew_param)?;
    let diff_w = w_calc
        .sub(weight_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_w < 1e-5,
        "Candle Cayley weight error {:.2e} exceeds 1e-5",
        diff_w
    );

    // Strict Orthogonality check: W^T @ W == I
    let wt = w_calc.t()?;
    let eye = Tensor::eye(3, DType::F32, &device)?;
    let ortho_check = wt.matmul(&w_calc)?;
    let diff_ortho = ortho_check
        .sub(&eye)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ortho < 1e-6,
        "Candle Orthogonality error {:.2e} exceeds 1e-6",
        diff_ortho
    );

    // 2. Isometry Norm Check: ||x @ W|| == ||x||
    let projected_x = test_in.matmul(&w_calc)?;
    let norm_in = (test_in.sqr()?.sum_keepdim(1)?.sqrt())?;
    let norm_out = (projected_x.sqr()?.sum_keepdim(1)?.sqrt())?;
    let diff_norm = norm_in
        .sub(&norm_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_norm < 1e-5,
        "Candle Isometry error {:.2e} exceeds 1e-5",
        diff_norm
    );

    // 3. Forward Output Parity
    let layer = OrthogonalLinear::new(skew_param.clone(), bias_gold.clone());
    let pred = layer.forward(test_in)?;
    let diff_pred = pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pred < 1e-5,
        "Candle forward prediction error {:.2e} exceeds 1e-5",
        diff_pred
    );

    // 4. Meta Device Simulation
    let meta = MetaTensor::new(&[8, 1024], DType::F32);
    let deep_meta = meta.propagate_deep_stack(2048, 50)?;
    assert_eq!(deep_meta.dims, vec![8, 2048]);
    assert_eq!(deep_meta.dtype, DType::F32);

    Ok(())
}
