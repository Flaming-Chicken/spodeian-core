use candle_core::{Device, Result, safetensors::load};

use ai_candle::c04_training_methods_and_optimisers::t16_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_16_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let z_emb = weights.get("z_emb").unwrap();
    let mrl_w = weights.get("mrl_w").unwrap();
    let mrl_b = weights.get("mrl_b").unwrap();
    let golden_logits_d2 = weights.get("golden_logits_d2").unwrap();
    let golden_logits_d4 = weights.get("golden_logits_d4").unwrap();
    let golden_logits_d8 = weights.get("golden_logits_d8").unwrap();

    let p_probs = weights.get("p_probs").unwrap();
    let y_targets = weights.get("y_targets").unwrap();
    let golden_el2n = weights.get("golden_el2n").unwrap();
    let loss_model = weights.get("loss_model").unwrap();
    let loss_ref = weights.get("loss_ref").unwrap();
    let golden_rho_loss = weights.get("golden_rho_loss").unwrap();

    let activations = weights.get("activations").unwrap();
    let golden_mean_act = weights.get("mean_act").unwrap();
    let golden_dormant_mask = weights.get("dormant_mask").unwrap();

    let anchor_w = weights.get("anchor_w").unwrap();
    let anchor_b = weights.get("anchor_b").unwrap();
    let anchor_x = weights.get("anchor_x").unwrap();
    let golden_anchor_out = weights.get("golden_anchor_out").unwrap();

    // 1. MRL Invariant
    let out_d2 = opt::mrl_slice_forward(z_emb, mrl_w, mrl_b, 2)?;
    let out_d4 = opt::mrl_slice_forward(z_emb, mrl_w, mrl_b, 4)?;
    let out_d8 = opt::mrl_slice_forward(z_emb, mrl_w, mrl_b, 8)?;

    let diff_d2 = out_d2
        .sub(golden_logits_d2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_d4 = out_d4
        .sub(golden_logits_d4)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_d8 = out_d8
        .sub(golden_logits_d8)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_d2 < 1e-5, "MRL d=2 diff: {}", diff_d2);
    assert!(diff_d4 < 1e-5, "MRL d=4 diff: {}", diff_d4);
    assert!(diff_d8 < 1e-5, "MRL d=8 diff: {}", diff_d8);

    // 2. EL2N Invariant
    let el2n = opt::compute_el2n(p_probs, y_targets)?;
    let diff_el2n = el2n
        .sub(golden_el2n)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_el2n < 1e-5, "EL2N diff: {}", diff_el2n);

    // 3. RHO-Loss Invariant
    let rho = opt::compute_rho_loss(loss_model, loss_ref)?;
    let diff_rho = rho
        .sub(golden_rho_loss)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_rho < 1e-5, "RHO diff: {}", diff_rho);

    // 4. ReDo Invariant
    let (mean_act, mask) = opt::detect_dormant_neurons(activations, 0.15)?;
    let diff_mean = mean_act
        .sub(golden_mean_act)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_mask = mask
        .sub(golden_dormant_mask)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_mean < 1e-5, "ReDo mean diff: {}", diff_mean);
    assert!(diff_mask < 1e-5, "ReDo mask diff: {}", diff_mask);

    // 5. Linear Anchor Invariant
    let anchor = opt::LinearAnchor::new(anchor_w.clone(), anchor_b.clone());
    let anchor_out = anchor.forward(anchor_x)?;
    let diff_anchor = anchor_out
        .sub(golden_anchor_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "Anchor diff: {}", diff_anchor);

    Ok(())
}
