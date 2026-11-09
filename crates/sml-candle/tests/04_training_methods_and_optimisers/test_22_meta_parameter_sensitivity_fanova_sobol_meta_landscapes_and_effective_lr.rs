use ai_candle::c04_training_methods_and_optimisers::t22_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr::MetaSensitivityUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_22_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr.safetensors");

    let module = MetaSensitivityUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Topic 22 Verified Errors: {:?}", errs);
    assert!(*errs.get("sobol_err").unwrap() < 1e-5);
    assert!(*errs.get("eff_lr_err").unwrap() < 1e-5);
    assert!(*errs.get("chaos_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
