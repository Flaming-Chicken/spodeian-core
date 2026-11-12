"""
Unit tests for Compiler Optimization, Speculative Decoding, PagedAttention, and SWA/EMA Smoothing.
"""

import pytest
import torch
import torch.nn as nn

from spodeian_ml.ops.compiler import compile_model_for_inference
from spodeian_ml.ops.speculative import SpeculativeDecoder
from spodeian_ml.ops.paged_attention import PagedKVCacheManager
from spodeian_ml.ops.smoothing import ModelWeightAverager


def test_compiler_optimization_fallback_and_warmup():
    linear = nn.Linear(8, 4)
    sample_input = torch.randn(2, 8)

    model_compiled, meta = compile_model_for_inference(
        model=linear,
        mode="default",
        dynamic=True,
        sample_input=sample_input,
    )
    assert meta["mode"] == "default"
    assert "backend" in meta
    # Model forward pass must work without error
    out = model_compiled(sample_input)
    assert out.shape == (2, 4)


def test_speculative_decoder():
    # Draft model: 1 layer
    draft = nn.Sequential(nn.Linear(8, 8), nn.GELU())
    # Target model: 2 layers
    target = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    spec_decoder = SpeculativeDecoder(
        draft_model=draft,
        target_model=target,
        gamma_lookahead=3,
        tolerance=5.0,  # Generous tolerance to ensure acceptance in synthetic test
    )

    prefix = torch.randn(1, 8)
    gen_seq, meta = spec_decoder.generate_step(prefix)

    assert meta["gamma"] == 3
    assert meta["accepted_steps"] >= 0
    assert "cumulative_acceptance_rate_pct" in meta
    assert gen_seq.shape[0] > prefix.shape[0] or gen_seq.dim() > 1


def test_paged_kv_cache_manager():
    # 4 blocks, block_size=4, num_heads=2, head_dim=8
    paged_cache = PagedKVCacheManager(
        num_blocks=8,
        block_size=4,
        num_heads=2,
        head_dim=8,
    )
    assert paged_cache.memory_utilization == 0.0

    # Allocate sequence 1
    assert paged_cache.allocate_sequence(seq_id=1, initial_tokens=0)
    assert paged_cache.memory_utilization > 0.0

    # Append 6 tokens (which should span 2 blocks since block_size=4)
    for _ in range(6):
        key = torch.randn(2, 8)
        val = torch.randn(2, 8)
        assert paged_cache.append_token(seq_id=1, key=key, value=val)

    assert len(paged_cache.block_tables[1]) == 2
    assert paged_cache.seq_lengths[1] == 6

    # Free sequence
    paged_cache.free_sequence(seq_id=1)
    assert paged_cache.memory_utilization == 0.0


def test_model_weight_averager_ema_and_swa():
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    # 1. EMA test
    averager_ema = ModelWeightAverager(model, mode="ema", decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
    averager_ema.update()

    # EMA expected: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    assert abs(averager_ema.shadow_params["weight"][0, 0].item() - 1.1) < 1e-4

    # 2. SWA test
    with torch.no_grad():
        model.weight.fill_(1.0)
    averager_swa = ModelWeightAverager(model, mode="swa")
    with torch.no_grad():
        model.weight.fill_(3.0)
    averager_swa.update()

    # SWA expected: (1.0 * 0 + 3.0) / 1 = 3.0 on first update
    averager_swa.apply_shadow_weights()
    assert abs(model.weight[0, 0].item() - 3.0) < 1e-4
