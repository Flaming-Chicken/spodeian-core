//! Integration test for Category 01 Topic 05: Parametric vs Non-Parametric & Neural Processes (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t05_parametric_vs_nonparametric_and_neural_processes::{
    AttentiveNeuralProcess, ConditionalNeuralProcess, ExactGaussianProcess, ParametricMLP,
};
use candle_core::{safetensors::load, Device, Result, Tensor};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("layers.weight").expect("layers.weight");
    let bias = tensors.get("layers.bias").expect("layers.bias");

    // 1. Canonical Linear Anchor Parity
    let linear_pred = test_in.matmul(weight)?.broadcast_add(bias)?;
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

    // 2. Parametric MLP Parity
    let x_target = tensors.get("layers.x_target").expect("x_target");
    let x_context = tensors.get("layers.x_context").expect("x_context");
    let y_context = tensors.get("layers.y_context").expect("y_context");

    let w_mlp1 = tensors.get("layers.w_mlp1").expect("w_mlp1");
    let b_mlp1 = tensors.get("layers.b_mlp1").expect("b_mlp1");
    let w_mlp2 = tensors.get("layers.w_mlp2").expect("w_mlp2");
    let b_mlp2 = tensors.get("layers.b_mlp2").expect("b_mlp2");
    let y_param_gold = tensors.get("layers.y_parametric").expect("y_parametric");

    let mlp = ParametricMLP::new(
        w_mlp1.clone(),
        b_mlp1.clone(),
        w_mlp2.clone(),
        b_mlp2.clone(),
    );
    let y_param_calc = mlp.forward(x_target)?;
    let diff_param = y_param_calc
        .sub(y_param_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_param < 1e-5,
        "Parametric MLP parity error {:.2e} exceeds 1e-5",
        diff_param
    );

    // 3. Exact Gaussian Process Parity
    let alpha = tensors.get("layers.alpha").expect("alpha");
    let gp_ls = tensors
        .get("layers.gp_lengthscale")
        .expect("gp_lengthscale");
    let ls_val = gp_ls.to_vec1::<f32>()?[0] as f64;
    let y_gp_gold = tensors.get("layers.y_gp").expect("y_gp");

    let gp = ExactGaussianProcess::new(alpha.clone(), ls_val);
    let y_gp_calc = gp.forward(x_target, x_context)?;
    let diff_gp = y_gp_calc
        .sub(y_gp_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_gp < 1e-5,
        "Exact GP parity error {:.2e} exceeds 1e-5",
        diff_gp
    );

    // 4. Attentive Neural Process Parity
    let w_enc = tensors.get("layers.w_enc").expect("w_enc");
    let b_enc = tensors.get("layers.b_enc").expect("b_enc");
    let w_q = tensors.get("layers.w_q").expect("w_q");
    let w_k = tensors.get("layers.w_k").expect("w_k");
    let w_v = tensors.get("layers.w_v").expect("w_v");
    let w_dec = tensors.get("layers.w_dec").expect("w_dec");
    let b_dec = tensors.get("layers.b_dec").expect("b_dec");

    let y_anp_gold = tensors.get("layers.y_anp").expect("y_anp");
    let attn_gold = tensors.get("layers.attn_weights").expect("attn_weights");

    let anp = AttentiveNeuralProcess::new(
        w_enc.clone(),
        b_enc.clone(),
        w_q.clone(),
        w_k.clone(),
        w_v.clone(),
        w_dec.clone(),
        b_dec.clone(),
    );
    let (y_anp_calc, attn_calc) = anp.forward(x_context, y_context, x_target)?;

    let diff_anp = y_anp_calc
        .sub(y_anp_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_anp < 1e-5,
        "ANP prediction parity error {:.2e} exceeds 1e-5",
        diff_anp
    );

    let diff_attn = attn_calc
        .sub(attn_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_attn < 1e-5,
        "ANP attention weights parity error {:.2e} exceeds 1e-5",
        diff_attn
    );

    // 5. Conditional Neural Process Parity
    let y_cnp_gold = tensors.get("layers.y_cnp").expect("y_cnp");
    let cnp =
        ConditionalNeuralProcess::new(w_enc.clone(), b_enc.clone(), w_dec.clone(), b_dec.clone());
    let y_cnp_calc = cnp.forward(x_context, y_context, x_target)?;
    let diff_cnp = y_cnp_calc
        .sub(y_cnp_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_cnp < 1e-5,
        "CNP prediction parity error {:.2e} exceeds 1e-5",
        diff_cnp
    );

    // 6. Attention Simplex Invariant: sum over context dim == 1.0
    let attn_sum = attn_calc.sum_keepdim(2)?;
    let ones = Tensor::ones_like(&attn_sum)?;
    let diff_simplex = attn_sum
        .sub(&ones)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_simplex < 1e-5,
        "Attention simplex invariant error {:.2e} exceeds 1e-5",
        diff_simplex
    );

    println!("Topic 05 Candle track 4-way transitive parity & mathematical invariants verified.");
    Ok(())
}
