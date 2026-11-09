use candle_core::{safetensors::load, Device, Result};

use ai_candle::c07_domain_tasks::t04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/04_nlp_tokenizers_bpe_wordpiece_unigram_and_byte_latent.safetensors",
        ),
        &device,
    )?;

    // 0. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. WordPiece Pair Score
    let wp = topic::wordpiece_pair_score(&w["bigram_counts"], &w["unigram_counts"], 1e-8)?;
    common::assert_max_diff(&wp, &w["wp_scores"], 1e-5)?;

    // 2. Byte Patch Pooling
    let patch = topic::byte_patch_pooling(&w["byte_embeds"], 4)?;
    common::assert_max_diff(&patch, &w["patch_embeds"], 1e-5)?;

    // 3. Dynamic Boundary Entropy
    let ent = topic::entropy_boundary_detector(&w["probs"], 1e-8)?;
    common::assert_max_diff(&ent, &w["entropy_ref"], 1e-5)?;

    Ok(())
}
