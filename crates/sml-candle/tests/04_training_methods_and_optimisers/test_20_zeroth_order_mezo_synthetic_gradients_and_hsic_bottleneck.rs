use ai_candle::c04_training_methods_and_optimisers::t20_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck::ZerothOrderUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_20_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.safetensors");

    let module = ZerothOrderUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Topic 20 Verified Errors: {:?}", errs);
    assert!(*errs.get("mezo_err").unwrap() < 1e-5);
    assert!(*errs.get("dni_err").unwrap() < 1e-5);
    assert!(*errs.get("hsic_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
