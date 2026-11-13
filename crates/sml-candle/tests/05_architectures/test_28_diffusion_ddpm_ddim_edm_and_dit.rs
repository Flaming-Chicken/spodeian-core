use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t28_diffusion_ddpm_ddim_edm_and_dit as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_28_diffusion_ddpm_ddim_edm_and_dit_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/05_generative_models/05_diffusion_ddpm_ddim_edm_and_dit.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. DDPM
    let x_t = topic::ddpm_forward_diffuse(&w["x_0"], &w["noise"], &w["alpha_bar_t"])?;
    common::assert_max_diff(&x_t, &w["x_t"], 1e-5)?;

    // 3. DDIM
    let (x_0_pred, x_prev) = topic::ddim_step(
        &w["x_t"],
        &w["eps_pred"],
        &w["alpha_bar_t"],
        &w["alpha_bar_prev"],
    )?;
    common::assert_max_diff(&x_0_pred, &w["x_0_pred"], 1e-5)?;
    common::assert_max_diff(&x_prev, &w["x_prev"], 1e-5)?;

    // 4. CFG
    let cfg_scale = w["cfg_scale"].to_vec1::<f32>()?[0] as f64;
    let cfg_out = topic::classifier_free_guidance(&w["eps_uncond"], &w["eps_cond"], cfg_scale)?;
    common::assert_max_diff(&cfg_out, &w["cfg_out"], 1e-5)?;

    // 5. DiT adaLN-Zero
    let adaln_out = topic::adaln_zero_modulation(
        &w["x_dit"],
        &w["gamma"],
        &w["beta_dit"],
        &w["alpha_dit"],
        1e-5,
    )?;
    common::assert_max_diff(&adaln_out, &w["adaln_out"], 1e-5)
}
