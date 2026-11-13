use candle_core::{Device, Result, safetensors::load};

use ai_candle::c17_deployment_low_level_and_capstones::t05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming.safetensors",
        ),
        &device,
    )?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let (y_prefill, k_cache, v_cache) = topic::prefill_step(
        &w["x_prefill"],
        &w["wq"],
        &w["wk"],
        &w["wv"],
        &w["wo"],
        &w["bo"],
    )?;
    common::assert_max_diff(&y_prefill, &w["y_prefill"], 1e-5)?;
    common::assert_max_diff(&k_cache, &w["k_cache"], 1e-5)?;
    common::assert_max_diff(&v_cache, &w["v_cache"], 1e-5)?;

    let (y_decode, new_k, new_v) = topic::decode_step(
        &w["x_decode"],
        &k_cache,
        &v_cache,
        &w["wq"],
        &w["wk"],
        &w["wv"],
        &w["wo"],
        &w["bo"],
    )?;
    common::assert_max_diff(&y_decode, &w["y_decode"], 1e-5)?;
    common::assert_max_diff(&new_k, &w["new_k"], 1e-5)?;
    common::assert_max_diff(&new_v, &w["new_v"], 1e-5)?;

    Ok(())
}
