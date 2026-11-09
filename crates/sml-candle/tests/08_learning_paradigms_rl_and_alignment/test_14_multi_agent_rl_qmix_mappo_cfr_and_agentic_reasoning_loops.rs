use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t14_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_14_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/07_multi_agent_rl_qmix_mappo_cfr_and_agentic_reasoning_loops.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. QMIX Monotonic Mixing
    let q_tot = topic::qmix_monotonic_mixing(
        &w["agent_qs"],
        &w["w1_raw"],
        &w["b1"],
        &w["w2_raw"],
        &w["b2"],
    )?;
    common::assert_max_diff(&q_tot, &w["qmix_q_tot_ref"], 1e-5)?;

    // 3. MAPPO Centralized GAE
    let adv = topic::mappo_centralized_gae(
        &w["values"],
        &w["rewards"],
        &w["dones"],
        &w["next_value"],
        0.99,
        0.95,
    )?;
    common::assert_max_diff(&adv, &w["mappo_advantages_ref"], 1e-5)?;

    // 4. CFR Regret Matching
    let strat = topic::cfr_regret_matching(&w["cumulative_regrets"], 1e-9)?;
    common::assert_max_diff(&strat, &w["cfr_strategy_ref"], 1e-5)?;

    // 5. Agentic Consensus Metric
    let consensus = topic::agentic_consensus_metric(&w["embeddings"], 1e-6)?;
    common::assert_max_diff(&consensus, &w["consensus_ref"], 1e-5)?;

    Ok(())
}
