use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t20_test_time_compute_prm_best_of_n_and_mcts as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_20_test_time_compute_prm_best_of_n_and_mcts_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/05_test_time_compute_prm_best_of_n_and_mcts.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. PRM product aggregation
    let prm_prod = topic::prm_product_aggregation(&w["step_probs"], &w["step_mask"], 1e-8)?;
    common::assert_max_diff(&prm_prod, &w["prm_prod_ref"], 1e-5)?;

    // 3. PRM min aggregation
    let prm_min = topic::prm_min_aggregation(&w["step_probs"], &w["step_mask"])?;
    common::assert_max_diff(&prm_min, &w["prm_min_ref"], 1e-5)?;

    // 4. Best-of-N reranking
    let (comb, best_idx) = topic::best_of_n_rerank(&w["orm_scores"], &w["prm_min_ref"], 0.5)?;
    common::assert_max_diff(&comb, &w["combined_scores_ref"], 1e-5)?;
    common::assert_max_diff(&best_idx, &w["best_idx_ref"], 1e-5)?;

    // 5. PUCT node selection
    let (puct_scores, selected_action) =
        topic::puct_node_selection(&w["q_values"], &w["priors"], &w["visit_counts"], 1.414)?;
    common::assert_max_diff(&puct_scores, &w["puct_scores_ref"], 1e-5)?;
    common::assert_max_diff(&selected_action, &w["selected_action_ref"], 1e-5)?;

    // 6. MCTS Backpropagation
    let action = w["selected_action_ref"].to_vec1::<f32>()?[0] as usize;
    let (q_up, v_up) =
        topic::mcts_backpropagation(&w["q_values"], &w["visit_counts"], action, 0.90)?;
    common::assert_max_diff(&q_up, &w["q_updated_ref"], 1e-5)?;
    common::assert_max_diff(&v_up, &w["visits_updated_ref"], 1e-5)?;

    Ok(())
}
