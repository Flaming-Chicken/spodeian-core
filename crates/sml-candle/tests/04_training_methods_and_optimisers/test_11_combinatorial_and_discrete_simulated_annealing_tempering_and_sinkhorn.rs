use candle_core::{Device, Result, safetensors::load};

use ai_candle::c04_training_methods_and_optimisers::t11_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let log_alpha = weights.get("log_alpha").unwrap();
    let candidates = weights.get("candidates").unwrap();
    let scores = weights.get("scores").unwrap();
    let delta_e = weights.get("delta_e").unwrap();
    let sa_temp = weights.get("sa_temp").unwrap().to_vec1::<f32>()?[0] as f64;
    let pt_e1 = weights.get("pt_e1").unwrap().to_vec1::<f32>()?[0] as f64;
    let pt_e2 = weights.get("pt_e2").unwrap().to_vec1::<f32>()?[0] as f64;
    let pt_t1 = weights.get("pt_t1").unwrap().to_vec1::<f32>()?[0] as f64;
    let pt_t2 = weights.get("pt_t2").unwrap().to_vec1::<f32>()?[0] as f64;

    let golden_sinkhorn_out = weights.get("golden_sinkhorn_out").unwrap();
    let golden_cem_mean = weights.get("golden_cem_mean").unwrap();
    let golden_cem_std = weights.get("golden_cem_std").unwrap();
    let golden_sa_probs = weights.get("golden_sa_probs").unwrap();
    let golden_pt_swap_prob = weights
        .get("golden_pt_swap_prob")
        .unwrap()
        .to_vec1::<f32>()?[0] as f64;

    // 1. Sinkhorn
    let sinkhorn_out = opt::run_sinkhorn(log_alpha, 10, 0.5)?;
    let diff_sinkhorn = sinkhorn_out
        .sub(golden_sinkhorn_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_sinkhorn < 1e-5, "Sinkhorn diff: {}", diff_sinkhorn);

    // 2. CEM
    let (cem_mean, cem_std) = opt::step_cem(candidates, scores, 0.25)?;
    let diff_mean = cem_mean
        .sub(golden_cem_mean)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_std = cem_std
        .sub(golden_cem_std)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_mean < 1e-5, "CEM mean diff: {}", diff_mean);
    assert!(diff_std < 1e-5, "CEM std diff: {}", diff_std);

    // 3. Simulated Annealing
    let sa_probs = opt::sa_acceptance_prob(delta_e, sa_temp)?;
    let diff_sa = sa_probs
        .sub(golden_sa_probs)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_sa < 1e-5, "SA diff: {}", diff_sa);

    // 4. Parallel Tempering
    let pt_prob = opt::parallel_tempering_swap_prob(pt_e1, pt_e2, pt_t1, pt_t2);
    let diff_pt = (pt_prob - golden_pt_swap_prob).abs();
    assert!(diff_pt < 1e-5, "PT diff: {}", diff_pt);

    // 5. Linear Anchor
    let anchor_w = weights.get("anchor_w").unwrap();
    let anchor_b = weights.get("anchor_b").unwrap();
    let anchor_x = weights.get("anchor_x").unwrap();
    let anchor = opt::LinearAnchor::new(anchor_w.clone(), anchor_b.clone());
    let anchor_out = anchor.forward(anchor_x)?;
    let expected_anchor = weights.get("golden_anchor_out").unwrap();
    let diff_anchor = anchor_out
        .sub(expected_anchor)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "anchor diff: {}", diff_anchor);

    // Unified module verification
    let module = opt::CombinatorialUnifiedModule::load_from_safetensors(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["sinkhorn_err"] < 1e-5);
    assert!(errs["cem_err"] < 1e-5);
    assert!(errs["sa_err"] < 1e-5);
    assert!(errs["pt_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
