use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t03_deepseek_moe_aux_loss_free_and_mixture_of_depths as ds;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_deepseek_moe_aux_loss_free_and_mixture_of_depths_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();
    let router_w = weights.get("router_weight").unwrap();
    let router_b = weights.get("router_bias").unwrap();
    let b_dyn = weights.get("b_dyn").unwrap();
    let shared_w1 = weights.get("shared_w1").unwrap();
    let shared_b1 = weights.get("shared_b1").unwrap();
    let shared_w2 = weights.get("shared_w2").unwrap();
    let shared_b2 = weights.get("shared_b2").unwrap();
    let routed_w1 = weights.get("routed_w1").unwrap();
    let routed_b1 = weights.get("routed_b1").unwrap();
    let routed_w2 = weights.get("routed_w2").unwrap();
    let routed_b2 = weights.get("routed_b2").unwrap();

    // 1. DeepSeekMoE Forward
    let (ds_out, counts) = ds::deepseek_moe_forward(
        x, router_w, router_b, b_dyn, shared_w1, shared_b1, shared_w2, shared_b2, routed_w1,
        routed_b1, routed_w2, routed_b2, 2,
    )?;
    let expected_ds = weights.get("deepseek_moe_output").unwrap();
    let diff_ds = ds_out
        .sub(expected_ds)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_ds < 1e-5, "DeepSeekMoE error: {}", diff_ds);

    // 2. Dynamic Bias Update
    let b_dyn_vec = b_dyn.to_vec1::<f32>()?;
    let b_updated = ds::update_dynamic_bias(&b_dyn_vec, &counts, 0.1);
    let expected_b = weights.get("b_dyn_updated").unwrap().to_vec1::<f32>()?;
    for (i, (&calc, &gold)) in b_updated.iter().zip(expected_b.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Dynamic bias mismatch at {i}: {calc} vs {gold}"
        );
    }

    // 3. Mixture of Depths (MoD) Forward
    let mod_w = weights.get("mod_weight").unwrap();
    let mod_b = weights.get("mod_bias").unwrap();
    let block_w1 = shared_w1.narrow(0, 0, 1)?.squeeze(0)?;
    let block_b1 = shared_b1.narrow(0, 0, 1)?.squeeze(0)?;
    let block_w2 = shared_w2.narrow(0, 0, 1)?.squeeze(0)?;
    let block_b2 = shared_b2.narrow(0, 0, 1)?.squeeze(0)?;

    let mod_out = ds::mixture_of_depths_forward(
        x, mod_w, mod_b, &block_w1, &block_b1, &block_w2, &block_b2, 2,
    )?;
    let expected_mod = weights.get("mod_output").unwrap();
    let diff_mod = mod_out
        .sub(expected_mod)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_mod < 1e-5, "Mixture of Depths error: {}", diff_mod);

    // 4. Anchor Parity
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let anchor_out = ds::linear_anchor_forward(test_in, weight, bias)?;
    let expected_anchor = weights.get("test_output").unwrap();
    let diff_anchor = anchor_out
        .sub(expected_anchor)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "Linear anchor error: {}", diff_anchor);

    Ok(())
}
