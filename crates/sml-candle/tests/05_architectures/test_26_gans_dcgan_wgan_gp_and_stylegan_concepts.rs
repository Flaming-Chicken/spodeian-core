use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t26_gans_dcgan_wgan_gp_and_stylegan_concepts as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_26_gans_dcgan_wgan_gp_and_stylegan_concepts_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. AdaIN
    let adain_out = topic::adain(&w["x_adain"], &w["scale"], &w["bias_adain"], 1e-5)?;
    common::assert_max_diff(&adain_out, &w["adain_out"], 1e-5)?;

    // 3. Weight Demodulation
    let demod_out =
        topic::stylegan2_weight_demodulation(&w["x_feat"], &w["w_conv"], &w["s_style"], 1e-8)?;
    common::assert_max_diff(&demod_out, &w["demod_out"], 1e-5)?;

    // 4. WGAN critic loss
    let wgan_loss = topic::wgan_critic_loss(&w["real_scores"], &w["fake_scores"])?.reshape((1,))?;
    common::assert_max_diff(&wgan_loss, &w["wgan_loss"], 1e-5)
}
