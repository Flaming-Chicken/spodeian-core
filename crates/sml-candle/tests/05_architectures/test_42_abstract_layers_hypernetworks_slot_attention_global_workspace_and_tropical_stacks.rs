use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t42_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_42_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks.safetensors",
        ),
        &device,
    )?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let y_hyp = topic::hypernetwork_linear(
        &w["x_hyper"],
        &w["z_code"],
        &w["w_h"],
        &w["b_h"],
        &w["bias_hyper"],
    )?;
    common::assert_max_diff(&y_hyp, &w["y_hyper"], 1e-5)?;

    let slots_out = topic::slot_attention(
        &w["slots_in"],
        &w["slot_inputs"],
        &w["w_q"],
        &w["w_k"],
        &w["w_v"],
    )?;
    common::assert_max_diff(&slots_out, &w["slots_out"], 1e-5)?;

    let y_trop = topic::tropical_max_plus_matmul(&w["x_trop"], &w["w_trop"])?;
    common::assert_max_diff(&y_trop, &w["y_trop"], 1e-5)?;

    Ok(())
}
