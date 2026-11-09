use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t31_discrete_diffusion_d3pm_mdlm_and_sedd as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_31_discrete_diffusion_d3pm_mdlm_and_sedd_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.safetensors"),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // D3PM forward marginal
    let x_t_uniform = topic::d3pm_forward_marginal(&w["x_0"], &w["q_uniform"])?;
    common::assert_max_diff(&x_t_uniform, &w["x_t_uniform"], 1e-5)?;

    // MDLM continuous-time marginal
    let x_t_mdlm = topic::mdlm_continuous_marginal(&w["x_0"], &w["alpha_t"], 4)?;
    common::assert_max_diff(&x_t_mdlm, &w["x_t_mdlm"], 1e-5)?;

    // Discrete denoiser
    let logits_pred = topic::discrete_denoiser(
        &x_t_mdlm,
        &w["t"],
        &w["w_x"],
        &w["b_x"],
        &w["w_t"],
        &w["b_t"],
        &w["w_out"],
        &w["b_out"],
    )?;
    common::assert_max_diff(&logits_pred, &w["logits_pred"], 1e-5)?;

    // Score ratio (softmax)
    let score_ratio = candle_nn::ops::softmax_last_dim(&logits_pred)?;
    common::assert_max_diff(&score_ratio, &w["score_ratio"], 1e-5)?;

    // SEDD score entropy loss
    let sedd_loss = topic::sedd_score_entropy(&score_ratio, &w["x_0"])?;
    common::assert_max_diff(&sedd_loss, &w["sedd_loss"], 1e-5)?;

    Ok(())
}
