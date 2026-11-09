use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t24_autoencoders_vae_beta_vae_and_vq_fsq_lfq as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_24_autoencoders_vae_beta_vae_and_vq_fsq_lfq_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("05_architectures/05_generative_models/01_autoencoders_vae_beta_vae_and_vq_fsq_lfq.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. VAE reparameterize & KL
    let beta: f64 = w["beta"].to_vec1::<f32>()?[0] as f64;
    let (z, kl_loss) = topic::vae_reparameterize_and_kl(&w["mu"], &w["logvar"], &w["eps"], beta)?;
    common::assert_max_diff(&z, &w["z"], 1e-5)?;
    common::assert_max_diff(&kl_loss, &w["kl_loss"], 1e-5)?;

    // 3. FSQ
    let half_width: f64 = w["half_width"].to_vec1::<f32>()?[0] as f64;
    let (z_q, fsq_indices) = topic::finite_scalar_quantization(&w["z_continuous"], half_width)?;
    common::assert_max_diff(&z_q, &w["z_q"], 1e-5)?;
    common::assert_max_diff(&fsq_indices, &w["fsq_indices"], 1e-5)
}
