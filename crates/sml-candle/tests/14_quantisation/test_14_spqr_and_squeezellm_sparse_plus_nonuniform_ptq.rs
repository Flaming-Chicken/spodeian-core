use candle_core::{Device, Result, safetensors::load};

use sml_candle::c14_quantisation::t14_spqr_and_squeezellm_sparse_plus_nonuniform_ptq as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_14_spqr_and_squeezellm_sparse_plus_nonuniform_ptq_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/03_llm_ptq_algorithms/07_spqr_and_squeezellm_sparse_plus_nonuniform_ptq.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Hessian and sensitivity
    let (h_diag, sensitivity) = topic::compute_hessian_sensitivity(&w["x_calib"], &w["w_orig"])?;
    common::assert_max_diff(&h_diag, &w["h_diag"], 1e-5)?;
    common::assert_max_diff(&sensitivity, &w["sensitivity"], 1e-5)?;

    // 3. Sparse extraction
    let (w_sparse, w_dense, sparse_mask) =
        topic::extract_sparse_outliers(&w["w_orig"], &sensitivity, 8)?;
    common::assert_max_diff(&w_sparse, &w["w_sparse"], 1e-5)?;
    common::assert_max_diff(&w_dense, &w["w_dense"], 1e-5)?;
    common::assert_max_diff(&sparse_mask, &w["sparse_mask"], 1e-5)?;

    // 4. Dense 3-bit quantization
    let (w_dense_deq, dense_scale) = topic::quantize_dense_int3(&w_dense, &sparse_mask, -4.0, 3.0)?;
    common::assert_max_diff(&w_dense_deq, &w["w_dense_deq"], 1e-5)?;
    common::assert_max_diff(&dense_scale, &w["dense_scale"], 1e-5)?;

    // 5. SpQR forward
    let (out_spqr, y_dense, y_sparse) =
        topic::spqr_forward(&w["x_calib"], &w_dense_deq, &w_sparse, Some(&w["b_orig"]))?;
    common::assert_max_diff(&out_spqr, &w["out_spqr"], 1e-5)?;
    common::assert_max_diff(&y_dense, &w["y_dense"], 1e-5)?;
    common::assert_max_diff(&y_sparse, &w["y_sparse"], 1e-5)?;

    Ok(())
}
