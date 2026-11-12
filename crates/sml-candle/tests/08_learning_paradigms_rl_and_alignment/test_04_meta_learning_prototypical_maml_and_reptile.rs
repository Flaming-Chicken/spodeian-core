use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t04_meta_learning_prototypical_maml_and_reptile as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_meta_learning_prototypical_maml_and_reptile_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. ProtoNet
    let prototypes = topic::protonet_prototypes(&w["support_embeds"])?;
    common::assert_max_diff(&prototypes, &w["prototypes_ref"], 1e-5)?;

    let logits = topic::protonet_logits(&w["query_embeds"], &prototypes)?;
    common::assert_max_diff(&logits, &w["protonet_logits_ref"], 1e-5)?;

    // 3. MAML
    let (w_ad, b_ad) =
        topic::maml_inner_linear_step(&w["x_s"], &w["y_s"], &w["w_maml"], &w["b_maml"], 0.05)?;
    common::assert_max_diff(&w_ad, &w["w_adapted_ref"], 1e-5)?;
    common::assert_max_diff(&b_ad, &w["b_adapted_ref"], 1e-5)?;

    // 4. Reptile
    let theta_new = topic::reptile_outer_step(&w["theta_base"], &w["theta_adapted"], 0.1)?;
    common::assert_max_diff(&theta_new, &w["theta_new_ref"], 1e-5)?;

    Ok(())
}
