use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t11_probabilistic_graphical_models_factor_graphs_bp_and_cavi as bp;
use std::path::PathBuf;

#[test]
fn test_11_probabilistic_graphical_models_factor_graphs_bp_and_cavi() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;

    // 2. Parity Check: Sum-Product Tree BP
    let unary_tree = tensors.get("unary_tree").expect("Missing unary_tree");
    let edge_tree = tensors.get("edge_tree").expect("Missing edge_tree");
    let golden_marg = tensors
        .get("exact_marginals")
        .expect("Missing exact_marginals");
    let golden_map = tensors.get("map_config").expect("Missing map_config");

    let marg = bp::sum_product_tree_bp(unary_tree, edge_tree)?;
    let diff_tree = (&marg - golden_marg)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_tree < 1e-4,
        "Candle vs Golden tree marginals error {diff_tree:.2e} exceeds 1e-4"
    );

    // 3. Parity Check: Max-Product Tree MAP
    let map_res = bp::max_product_tree_map(unary_tree, edge_tree)?;
    let map_diff = (&map_res - golden_map)?.abs()?.max(0)?.to_scalar::<i64>()?;
    assert_eq!(map_diff, 0, "Candle vs Golden MAP assignment mismatch");

    // 4. Parity Check: Damped Loopy BP
    let unary_tri = tensors.get("unary_tri").expect("Missing unary_tri");
    let edge_tri = tensors.get("edge_tri").expect("Missing edge_tri");
    let golden_loopy = tensors
        .get("loopy_marginals")
        .expect("Missing loopy_marginals");

    let loopy_res = bp::damped_loopy_bp_triangle(unary_tri, edge_tri, 0.5, 20)?;
    let diff_loopy = (&loopy_res - golden_loopy)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_loopy < 1e-4,
        "Candle vs Golden Loopy BP marginals error {diff_loopy:.2e} exceeds 1e-4"
    );

    // 5. Parity Check: CAVI
    let data_cavi = tensors.get("data_cavi").expect("Missing data_cavi");
    let mu_cavi = tensors.get("mu_cavi").expect("Missing mu_cavi");
    let golden_resp = tensors
        .get("responsibilities")
        .expect("Missing responsibilities");
    let golden_elbo = tensors.get("elbo_val").expect("Missing elbo_val");

    let (resp, elbo) = bp::cavi_gmm_step(data_cavi, mu_cavi, 1.0)?;
    let diff_resp = (&resp - golden_resp)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_elbo = (&elbo - golden_elbo)?.abs()?.max(0)?.to_scalar::<f32>()?;

    assert!(
        diff_resp < 1e-4,
        "Candle vs Golden CAVI responsibilities error {diff_resp:.2e} exceeds 1e-4"
    );
    assert!(
        diff_elbo < 1e-4,
        "Candle vs Golden CAVI ELBO error {diff_elbo:.2e} exceeds 1e-4"
    );

    println!("Candle Topic 11: Tree BP, Loopy BP & CAVI tests passed with machine precision!");
    Ok(())
}
