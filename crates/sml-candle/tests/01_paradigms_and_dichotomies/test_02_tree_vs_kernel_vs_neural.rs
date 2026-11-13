//! Integration test for Category 01 Topic 02: Tree vs Kernel vs Neural Inductive Biases (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t02_tree_vs_kernel_vs_neural_inductive_biases::{
    MLPClassifier, RBFKernelRidge, SoftDecisionTree,
};
use candle_core::{Device, Result, Tensor, safetensors::load};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.safetensors",
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

    // 2. Soft Decision Tree Parity
    let tree_route = tensors
        .get("layers.tree_routing_weights")
        .expect("tree_routing_weights");
    let tree_thresh = tensors
        .get("layers.tree_thresholds")
        .expect("tree_thresholds");
    let tree_temp = tensors
        .get("layers.tree_temperature")
        .expect("tree_temperature")
        .to_vec1::<f32>()?[0] as f64;
    let tree_leaf = tensors
        .get("layers.tree_leaf_weights")
        .expect("tree_leaf_weights");
    let tree_gold = tensors.get("layers.tree_output").expect("tree_output");

    let tree_model = SoftDecisionTree::new(
        tree_route.clone(),
        tree_thresh.clone(),
        tree_temp,
        tree_leaf.clone(),
    );
    let tree_pred = tree_model.forward(test_in)?;
    let diff_tree = tree_pred
        .sub(tree_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_tree < 1e-5,
        "Soft Decision Tree parity error {:.2e} exceeds 1e-5",
        diff_tree
    );

    // 3. RBF Kernel Ridge Parity
    let kernel_supp = tensors
        .get("layers.kernel_support")
        .expect("kernel_support");
    let kernel_coeffs = tensors
        .get("layers.kernel_dual_coeffs")
        .expect("kernel_dual_coeffs");
    let kernel_gamma = tensors
        .get("layers.kernel_gamma")
        .expect("kernel_gamma")
        .to_vec1::<f32>()?[0] as f64;
    let kernel_bias = tensors.get("layers.kernel_bias").expect("kernel_bias");
    let kernel_gold = tensors.get("layers.kernel_output").expect("kernel_output");

    let kernel_model = RBFKernelRidge::new(
        kernel_supp.clone(),
        kernel_coeffs.clone(),
        kernel_gamma,
        kernel_bias.clone(),
    );
    let kernel_pred = kernel_model.forward(test_in)?;
    let diff_kernel = kernel_pred
        .sub(kernel_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_kernel < 1e-5,
        "RBF Kernel Ridge parity error {:.2e} exceeds 1e-5",
        diff_kernel
    );

    // 4. MLP Classifier Parity
    let mlp_w1 = tensors.get("layers.mlp_w1").expect("mlp_w1");
    let mlp_b1 = tensors.get("layers.mlp_b1").expect("mlp_b1");
    let mlp_w2 = tensors.get("layers.mlp_w2").expect("mlp_w2");
    let mlp_b2 = tensors.get("layers.mlp_b2").expect("mlp_b2");
    let mlp_gold = tensors.get("layers.mlp_output").expect("mlp_output");

    let mlp_model = MLPClassifier::new(
        mlp_w1.clone(),
        mlp_b1.clone(),
        mlp_w2.clone(),
        mlp_b2.clone(),
    );
    let mlp_pred = mlp_model.forward(test_in)?;
    let diff_mlp = mlp_pred
        .sub(mlp_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_mlp < 1e-5,
        "MLP Classifier parity error {:.2e} exceeds 1e-5",
        diff_mlp
    );

    // 5. Invariant Test: RBF Kernel Orthogonal Rotation Equivariance
    // For orthogonal rotation matrix R (e.g. 2D rotation block), ||R x - R s||^2 == ||x - s||^2
    let angle = std::f32::consts::FRAC_PI_4; // 45 degrees
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    // 4x4 block rotation on first 2 coords
    let rot_matrix = Tensor::new(
        &[
            [cos_a, -sin_a, 0.0f32, 0.0],
            [sin_a, cos_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        &device,
    )?;

    let rot_in = test_in.matmul(&rot_matrix)?;
    let rot_supp = kernel_supp.matmul(&rot_matrix)?;
    let rot_kernel_model = RBFKernelRidge::new(
        rot_supp,
        kernel_coeffs.clone(),
        kernel_gamma,
        kernel_bias.clone(),
    );
    let rot_pred = rot_kernel_model.forward(&rot_in)?;
    let rot_diff = rot_pred
        .sub(&kernel_pred)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        rot_diff < 1e-5,
        "RBF Kernel rotation equivariance error {:.2e} exceeds 1e-5",
        rot_diff
    );

    Ok(())
}
