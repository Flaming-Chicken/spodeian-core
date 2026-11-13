use candle_core::{Device, Result, safetensors::load};

use sml_candle::c04_training_methods_and_optimisers::t06_schedule_free_and_parameter_free_optimisers as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let sf_x = weights.get("sf_x").unwrap();
    let sf_z = weights.get("sf_z").unwrap();
    let sf_y = weights.get("sf_y").unwrap();
    let sf_grad = weights.get("sf_grad").unwrap();
    let sf_m = weights.get("sf_m").unwrap();
    let sf_v = weights.get("sf_v").unwrap();

    let polyak_param = weights.get("polyak_param").unwrap();
    let polyak_grad = weights.get("polyak_grad").unwrap();
    let f_val = weights.get("f_val").unwrap().to_vec1::<f32>()?[0] as f64;
    let f_opt = weights.get("f_opt").unwrap().to_vec1::<f32>()?[0] as f64;

    let bb_param = weights.get("bb_param").unwrap();
    let bb_grad = weights.get("bb_grad").unwrap();
    let bb_s = weights.get("bb_s").unwrap();
    let bb_y = weights.get("bb_y").unwrap();

    let d_param = weights.get("d_param").unwrap();
    let d_grad = weights.get("d_grad").unwrap();
    let d_est = weights.get("d_est").unwrap().to_vec1::<f32>()?[0] as f64;

    // 1. Schedule-Free AdamW
    let (x_next, z_next, y_next, _, _) = opt::step_schedule_free_adamw(
        sf_x, sf_z, sf_y, sf_grad, sf_m, sf_v, 0.01, 0.9, 0.999, 0.01, 1, 1e-8,
    )?;
    let expected_x = weights.get("golden_sf_x").unwrap();
    let expected_z = weights.get("golden_sf_z").unwrap();
    let expected_y = weights.get("golden_sf_y").unwrap();
    let diff_x = x_next
        .sub(expected_x)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_z = z_next
        .sub(expected_z)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_y = y_next
        .sub(expected_y)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_x < 1e-5, "sf_x diff: {}", diff_x);
    assert!(diff_z < 1e-5, "sf_z diff: {}", diff_z);
    assert!(diff_y < 1e-5, "sf_y diff: {}", diff_y);

    // 2. Polyak Step-Size
    let p_polyak = opt::step_polyak(polyak_param, polyak_grad, f_val, f_opt, 1e-8)?;
    let expected_polyak = weights.get("golden_polyak_param").unwrap();
    let diff_polyak = p_polyak
        .sub(expected_polyak)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_polyak < 1e-5, "polyak diff: {}", diff_polyak);

    // 3. Barzilai-Borwein BB1
    let p_bb = opt::step_barzilai_borwein(bb_param, bb_grad, bb_s, bb_y, 1e-8)?;
    let expected_bb = weights.get("golden_bb_param").unwrap();
    let diff_bb = p_bb
        .sub(expected_bb)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_bb < 1e-5, "bb diff: {}", diff_bb);

    // 4. D-Adaptation
    let (p_dadapt, _) = opt::step_d_adaptation(d_param, d_grad, d_est, 0.01, 1e-8)?;
    let expected_dadapt = weights.get("golden_d_adapt_param").unwrap();
    let diff_dadapt = p_dadapt
        .sub(expected_dadapt)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_dadapt < 1e-5, "dadapt diff: {}", diff_dadapt);

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
    let module = opt::ScheduleFreeUnifiedModule::load_from_safetensors(&weights_path, &device)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["schedule_free_err"] < 1e-5);
    assert!(errs["polyak_err"] < 1e-5);
    assert!(errs["bb_err"] < 1e-5);
    assert!(errs["d_adapt_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
