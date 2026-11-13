use candle_core::{Device, Result, safetensors::load};

use sml_candle::c07_domain_tasks::t05_nlp_sequence_tagging_bilstm_crf_and_span_ner as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_nlp_sequence_tagging_bilstm_crf_and_span_ner_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/05_nlp_sequence_tagging_bilstm_crf_and_span_ner.safetensors",
        ),
        &device,
    )?;

    // 0. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. CRF log Z
    let log_z = topic::crf_log_partition(&w["emissions"], &w["transitions"])?;
    common::assert_max_diff(&log_z, &w["log_z"], 1e-5)?;

    // 2. CRF Gold Sequence Score
    let gold_score = topic::crf_sequence_score(&w["emissions"], &w["transitions"], &w["tags"])?;
    common::assert_max_diff(&gold_score, &w["gold_score"], 1e-5)?;

    // 3. CRF NLL Loss
    let loss = topic::crf_nll_loss(&w["emissions"], &w["transitions"], &w["tags"])?;
    common::assert_max_diff(&loss, &w["crf_loss"], 1e-5)?;

    // 4. Span Endpoint Features
    let span_feats =
        topic::span_endpoint_features(&w["seq_embeds"], &w["start_idx"], &w["end_idx"])?;
    common::assert_max_diff(&span_feats, &w["span_feats"], 1e-5)?;

    Ok(())
}
