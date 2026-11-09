use ai_candle::c04_training_methods_and_optimisers::t23_curriculum_self_paced_learning_and_dataset_distillation_mtt::CurriculumDistillationUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_23_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/09_curriculum_self_paced_learning_and_dataset_distillation_mtt.safetensors");

    let module = CurriculumDistillationUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Topic 23 Verified Errors: {:?}", errs);
    assert!(*errs.get("spl_err").unwrap() < 1e-5);
    assert!(*errs.get("curriculum_err").unwrap() < 1e-5);
    assert!(*errs.get("gm_err").unwrap() < 1e-5);
    assert!(*errs.get("mtt_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
