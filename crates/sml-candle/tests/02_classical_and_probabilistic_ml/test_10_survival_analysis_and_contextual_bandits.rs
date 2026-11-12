use ai_candle::c02_classical_and_probabilistic_ml::t10_survival_analysis_and_contextual_bandits as surv;
use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;

#[test]
fn test_10_survival_analysis_and_contextual_bandits() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;

    // 2. Parity Check: Cox Proportional Hazards Partial Log-Likelihood
    let x_surv = tensors.get("X_surv").expect("Missing X_surv");
    let t_surv = tensors.get("T_surv").expect("Missing T_surv");
    let e_surv = tensors.get("E_surv").expect("Missing E_surv");
    let w_surv = tensors.get("w_surv").expect("Missing w_surv");
    let b_surv = tensors.get("b_surv").expect("Missing b_surv");
    let golden_risk = tensors.get("risk_scores").expect("Missing risk_scores");
    let golden_cox_loss = tensors.get("cox_loss").expect("Missing cox_loss");

    let w_col = w_surv.unsqueeze(1)?;
    let xw = x_surv.matmul(&w_col)?.squeeze(1)?;
    let risk = (xw + b_surv.broadcast_as(t_surv.shape())?)?;

    let diff_risk = (&risk - golden_risk)?.abs()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        diff_risk < 1e-4,
        "Candle vs Golden risk score error {diff_risk:.2e} exceeds 1e-4"
    );

    let cox_loss = surv::cox_ph_loss(&risk, t_surv, e_surv)?;
    let diff_loss = (&cox_loss - golden_cox_loss)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_loss < 1e-4,
        "Candle vs Golden Cox partial loss error {diff_loss:.2e} exceeds 1e-4"
    );

    // 3. Parity Check: Kaplan-Meier Survival Curve
    let golden_km_times = tensors.get("km_times").expect("Missing km_times");
    let golden_km_surv = tensors.get("km_survival").expect("Missing km_survival");

    let (km_times, km_surv) = surv::kaplan_meier_survival(t_surv, e_surv)?;
    let diff_km_t = (&km_times - golden_km_times)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_km_s = (&km_surv - golden_km_surv)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(
        diff_km_t < 1e-4,
        "Candle vs Golden KM times error {diff_km_t:.2e} exceeds 1e-4"
    );
    assert!(
        diff_km_s < 1e-4,
        "Candle vs Golden KM survival error {diff_km_s:.2e} exceeds 1e-4"
    );

    // 4. Parity Check: LinUCB Contextual Bandits
    let a_arms = tensors.get("A_arms").expect("Missing A_arms");
    let b_arms = tensors.get("b_arms").expect("Missing b_arms");
    let x_ctx = tensors.get("x_context").expect("Missing x_context");
    let golden_scores = tensors.get("linucb_scores").expect("Missing linucb_scores");
    let golden_arm = tensors.get("selected_arm").expect("Missing selected_arm");

    let (scores, selected) = surv::linucb_score_arms(x_ctx, a_arms, b_arms, 1.0)?;
    let diff_scores = (&scores - golden_scores)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    let arm_diff = (&selected - golden_arm)?
        .abs()?
        .max(0)?
        .to_scalar::<i64>()?;

    assert!(
        diff_scores < 1e-4,
        "Candle vs Golden LinUCB scores error {diff_scores:.2e} exceeds 1e-4"
    );
    assert_eq!(arm_diff, 0, "Candle vs Golden selected arm mismatch");

    // 5. Invariant Check: Sherman-Morrison Update Exact Inversion
    let a0 = a_arms.get(0)?;
    let a0_slice: Vec<f32> = a0.flatten_all()?.to_vec1()?;
    let a0_inv_vec = surv::invert_matrix_cpu(&a0_slice, 4).unwrap();
    let a0_inv = Tensor::from_vec(a0_inv_vec, (4, 4), &device)?;

    let a_inv_updated = surv::sherman_morrison_rank1_update(&a0_inv, x_ctx)?;

    let x_col = x_ctx.unsqueeze(1)?;
    let x_row = x_ctx.unsqueeze(0)?;
    let a_exact = (&a0 + x_col.matmul(&x_row)?)?;
    let a_exact_slice: Vec<f32> = a_exact.flatten_all()?.to_vec1()?;
    let a_exact_inv_vec = surv::invert_matrix_cpu(&a_exact_slice, 4).unwrap();
    let a_exact_inv = Tensor::from_vec(a_exact_inv_vec, (4, 4), &device)?;

    let diff_sm = (&a_inv_updated - &a_exact_inv)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sm < 1e-4,
        "Sherman-Morrison inverse update error {diff_sm:.2e} exceeds 1e-4"
    );

    println!("Candle Topic 10: Cox, KM & LinUCB tests passed with machine precision!");
    Ok(())
}
