use ai_candle::c04_training_methods_and_optimisers::t18_self_optimising_test_time_adaptation_tent_memo_star_and_spin::TestTimeAdaptationUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_18_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.safetensors");

    let module = TestTimeAdaptationUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Topic 18 Verified Errors: {:?}", errs);
    assert!(*errs.get("tent_err").unwrap() < 1e-5);
    assert!(*errs.get("memo_err").unwrap() < 1e-5);
    assert!(*errs.get("spin_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
