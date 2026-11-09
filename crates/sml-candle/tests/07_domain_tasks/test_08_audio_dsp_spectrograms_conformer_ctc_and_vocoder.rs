use candle_core::{safetensors::load, Device, Result};

use ai_candle::c07_domain_tasks::t08_audio_dsp_spectrograms_conformer_ctc_and_vocoder as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_audio_dsp_spectrograms_conformer_ctc_and_vocoder_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/08_audio_dsp_spectrograms_conformer_ctc_and_vocoder.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Log Mel Spectrogram
    let log_mel = topic::log_mel_spectrogram(&w["magnitude_spec"], &w["mel_basis"], 1e-5)?;
    common::assert_max_diff(&log_mel, &w["log_mel_ref"], 1e-5)?;

    // 2. Conformer Conv Module
    let conf_out = topic::conformer_conv_module(
        &w["conf_x"],
        &w["w_p1"],
        &w["b_p1"],
        &w["w_dw"],
        &w["b_dw"],
        &w["w_p2"],
        &w["b_p2"],
    )?;
    common::assert_max_diff(&conf_out, &w["conf_out_ref"], 1e-5)?;

    // 3. CTC Loss
    let target = vec![0u32, 2u32];
    let ctc_loss = topic::ctc_forward_loss(&w["ctc_log_probs"], &target, 3)?;
    common::assert_max_diff(&ctc_loss, &w["ctc_loss_ref"], 1e-5)?;

    // 4. MR-STFT Loss
    let mrstft = topic::mrstft_loss(&w["y_pred"], &w["y_true"], 1e-5)?;
    common::assert_max_diff(&mrstft, &w["mrstft_ref"], 1e-5)?;

    Ok(())
}
