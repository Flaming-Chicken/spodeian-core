use candle_core::{Device, Result, Tensor, safetensors::load};

use sml_candle::c06_smooth_and_physics_informed_ai::t11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // LISTA unroll
    let z_lista = topic::lista_unroll(&w["y_meas"], &w["w_e"], &w["w_s"], &w["theta_lista"], 3)?;
    common::assert_max_diff(&z_lista, &w["z_lista"], 1e-5)?;

    // ADMM-Net step
    let (b_sz, n) = w["x1"].dims2()?;
    let x0 = Tensor::zeros((b_sz, n), candle_core::DType::F32, &device)?;
    let z0 = Tensor::zeros((b_sz, n), candle_core::DType::F32, &device)?;
    let u0 = Tensor::zeros((b_sz, n), candle_core::DType::F32, &device)?;

    let (x1, z1, u1) = topic::admm_net_step(&w["y_admm"], &x0, &z0, &u0, &w["w_inv"], 1.0, 0.08)?;
    common::assert_max_diff(&x1, &w["x1"], 1e-5)?;
    common::assert_max_diff(&z1, &w["z1"], 1e-5)?;
    common::assert_max_diff(&u1, &w["u1"], 1e-5)?;

    // CRATE step
    let z_crate = topic::crate_step(&w["z_crate_in"], &w["d_dict"], 0.1, 0.04)?;
    common::assert_max_diff(&z_crate, &w["z_crate_out"], 1e-5)?;

    Ok(())
}
