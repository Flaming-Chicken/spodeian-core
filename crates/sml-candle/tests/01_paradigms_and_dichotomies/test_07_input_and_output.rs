//! Integration test for Category 01 Topic 07: Input & Output Spaces (Candle Track).

use sml_candle::c01_paradigms_and_dichotomies::t07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds::{
    fourier_coordinate_encode, rotation_6d_to_matrix, PatchEmbedding, PointerNetworkLayer,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.safetensors",
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

    // 2. Continuous 6D Rotation Manifold Parity
    let rot_gold = tensors.get("layers.rot_matrices").expect("rot_matrices");
    let rot_calc = rotation_6d_to_matrix(test_in)?;
    let diff_rot = rot_calc
        .sub(rot_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rot < 1e-5,
        "6D rotation manifold parity error {:.2e} exceeds 1e-5",
        diff_rot
    );

    // Invariant: R^T R = I (orthonormality)
    let rot_t = rot_calc.transpose(1, 2)?;
    let r_rt = rot_t.matmul(&rot_calc)?;
    let b = test_in.dim(0)?;
    let eye_3d =
        candle_core::Tensor::eye(3, candle_core::DType::F32, &device)?.broadcast_as((b, 3, 3))?;
    let diff_ortho = r_rt
        .sub(&eye_3d)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ortho < 1e-5,
        "Rotation matrix orthonormality error {:.2e} exceeds 1e-5",
        diff_ortho
    );

    // 3. Pointer Network Parity
    let enc_seq = tensors.get("layers.encoder_seq").expect("encoder_seq");
    let dec_state = tensors.get("layers.decoder_state").expect("decoder_state");
    let w_enc = tensors.get("layers.w_enc").expect("w_enc");
    let w_dec = tensors.get("layers.w_dec").expect("w_dec");
    let b_ptr = tensors.get("layers.b_ptr").expect("b_ptr");
    let v_ptr = tensors.get("layers.v_ptr").expect("v_ptr");
    let ptr_logits_gold = tensors.get("layers.ptr_logits").expect("ptr_logits");
    let ptr_probs_gold = tensors.get("layers.ptr_probs").expect("ptr_probs");

    let ptr_layer =
        PointerNetworkLayer::new(w_enc.clone(), w_dec.clone(), b_ptr.clone(), v_ptr.clone());
    let (logits_calc, probs_calc) = ptr_layer.forward(enc_seq, dec_state)?;
    let diff_logits = logits_calc
        .sub(ptr_logits_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_logits < 1e-5,
        "Pointer logits parity error {:.2e} exceeds 1e-5",
        diff_logits
    );

    let diff_probs = probs_calc
        .sub(ptr_probs_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_probs < 1e-5,
        "Pointer probs parity error {:.2e} exceeds 1e-5",
        diff_probs
    );

    // Invariant: Pointer probabilities sum to 1.0 along last dimension
    let prob_sums = probs_calc.sum_keepdim(1)?.squeeze(1)?;
    let ones = candle_core::Tensor::ones(b, candle_core::DType::F32, &device)?;
    let diff_sum = prob_sums.sub(&ones)?.abs()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        diff_sum < 1e-5,
        "Pointer probability simplex sum error {:.2e} exceeds 1e-5",
        diff_sum
    );

    // 4. Multi-Scale Fourier Coordinate Encoding Parity
    let coords_2d = tensors.get("layers.coords_2d").expect("coords_2d");
    let coords_fourier_gold = tensors
        .get("layers.coords_fourier")
        .expect("coords_fourier");
    let fourier_calc = fourier_coordinate_encode(coords_2d, 4)?;
    let diff_fourier = fourier_calc
        .sub(coords_fourier_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_fourier < 1e-5,
        "Fourier coordinate encoding parity error {:.2e} exceeds 1e-5",
        diff_fourier
    );

    // 5. ViT Patch Linear Embedding Parity
    let patches = tensors.get("layers.patches").expect("patches");
    let w_patch = tensors.get("layers.w_patch").expect("w_patch");
    let b_patch = tensors.get("layers.b_patch").expect("b_patch");
    let patch_embed_gold = tensors
        .get("layers.patch_embeddings")
        .expect("patch_embeddings");

    let patch_layer = PatchEmbedding::new(w_patch.clone(), b_patch.clone());
    let patch_embed_calc = patch_layer.forward(patches)?;
    let diff_patches = patch_embed_calc
        .sub(patch_embed_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_patches < 1e-5,
        "Patch embedding parity error {:.2e} exceeds 1e-5",
        diff_patches
    );

    Ok(())
}
