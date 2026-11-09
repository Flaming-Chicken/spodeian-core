use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t41_spiking_neural_networks_lif_surrogate_gradients_and_capsules as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_41_spiking_neural_networks_lif_surrogate_gradients_and_capsules_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.safetensors"), &device)?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let (spikes, voltages) = topic::lif_simulation(&w["currents"], 0.8, 1.0)?;
    common::assert_max_diff(&spikes, &w["lif_spikes"], 1e-5)?;
    common::assert_max_diff(&voltages, &w["lif_voltages"], 1e-5)?;

    let v_caps = topic::capsule_dynamic_routing(&w["u_hat"], 3)?;
    common::assert_max_diff(&v_caps, &w["v_capsules"], 1e-5)?;

    Ok(())
}
