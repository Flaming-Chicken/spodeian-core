use ai_candle::c02_classical_and_probabilistic_ml::t01_generalized_linear_models_and_proximal_lasso::GLMUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_01_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.safetensors");

    let module = GLMUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Category 02 Topic 01 Verified Errors: {:?}", errs);
    assert!(*errs.get("ols_err").unwrap() < 1e-5);
    assert!(*errs.get("ridge_err").unwrap() < 1e-5);
    assert!(*errs.get("lasso_err").unwrap() < 1e-5);
    assert!(*errs.get("poisson_err").unwrap() < 1e-5);
    assert!(*errs.get("huber_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
