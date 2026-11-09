use candle_core::{safetensors::load, Device, Result};

use ai_candle::c07_domain_tasks::t10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. FT-Transformer Tokenizer
    let tokens = topic::ft_transformer_tokenize(
        &w["x_num"],
        &w["w_num"],
        &w["b_num"],
        &w["cat_tokens"],
        &w["cls_token"],
    )?;
    common::assert_max_diff(&tokens, &w["tokenized_ref"], 1e-5)?;

    // 2. Sparsemax
    let sm = topic::sparsemax(&w["z_sparsemax"])?;
    common::assert_max_diff(&sm, &w["sparsemax_ref"], 1e-5)?;

    // 3. TabNet Attentive Mask
    let (mask, p_next) = topic::tabnet_attentive_mask(&w["p_prior"], &w["h_feat"], 1.3)?;
    common::assert_max_diff(&mask, &w["tabnet_mask_ref"], 1e-5)?;
    common::assert_max_diff(&p_next, &w["tabnet_p_next_ref"], 1e-5)?;

    // 4. PLE Task Routing
    let indices = vec![0u32, 1u32, 4u32, 5u32];
    let ple = topic::ple_task_routing(&w["expert_outputs"], &w["gate_logits"], &indices)?;
    common::assert_max_diff(&ple, &w["ple_routed_ref"], 1e-5)?;

    Ok(())
}
