use candle_core::{Device, Result, safetensors::load};

use ai_candle::c14_quantisation::t02_logarithmic_pot_apot_mulaw_and_posit_quantization as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_logarithmic_pot_apot_mulaw_and_posit_quantization_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. PoT Quantization
    let pot = topic::pot_quantize(&w["raw_pot"], 4)?;
    common::assert_max_diff(&pot, &w["pot_ref"], 1e-5)?;

    // 3. APoT Quantization
    let apot = topic::apot_quantize(&w["raw_apot"])?;
    common::assert_max_diff(&apot, &w["apot_ref"], 1e-5)?;

    // 4. Mu-Law Compand-Expand
    let mulaw = topic::mu_law_compand_expand(&w["raw_audio"], 255.0, 8)?;
    common::assert_max_diff(&mulaw, &w["mulaw_ref"], 1e-5)?;

    // 5. Base-sqrt(2) Log Quantization
    let sqrt2 = topic::base_sqrt2_log_quantize(&w["raw_sqrt2"])?;
    common::assert_max_diff(&sqrt2, &w["sqrt2_ref"], 1e-5)?;

    // 6. Posit4 Quantization
    let posit4 = topic::posit4_quantize(&w["raw_posit"])?;
    common::assert_max_diff(&posit4, &w["posit4_ref"], 1e-5)?;

    Ok(())
}
