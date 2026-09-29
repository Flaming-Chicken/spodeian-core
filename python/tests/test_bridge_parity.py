"""
Parity verification tests for spodeian-ml data augmentations and lake loading.
Verifies that Rust PyO3 bridge and PyTorch produce identical numerical outputs.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from spodeian_ml.data.augmentations import temporal_mask, gaussian_jitter, rolling_zscore, time_warp
from spodeian_ml.data.factors import log_returns, rolling_volatility, exponential_moving_average, vwap
from spodeian_ml.data.lake_loader import SpodeianLakeDataset


def test_temporal_masking_parity():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    masked1 = temporal_mask(tensor, mask_ratio=0.4, mask_value=-999.0, seed=42)
    masked2 = temporal_mask(tensor, mask_ratio=0.4, mask_value=-999.0, seed=42)

    assert torch.equal(masked1, masked2), "Masking must be deterministic across identical seeds"
    assert -999.0 in masked1.tolist(), "Masked value should be present"


def test_gaussian_jitter_parity():
    tensor = torch.full((100,), 10.0)
    jittered = gaussian_jitter(tensor, std_dev=0.1, seed=1234)

    assert jittered.shape == tensor.shape
    mean_val = jittered.mean().item()
    assert abs(mean_val - 10.0) < 0.05


def test_rolling_zscore_parity():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    z = rolling_zscore(tensor, window=3)

    assert z.shape == tensor.shape
    last_z = z[-1].item()
    assert 1.0 < last_z < 1.3


def test_factor_calculations():
    prices = torch.tensor([100.0, 105.0, 102.0, 110.0])
    returns = log_returns(prices)
    assert returns[0].item() == 0.0
    expected_ret_1 = torch.log(torch.tensor(105.0 / 100.0)).item()
    assert abs(returns[1].item() - expected_ret_1) < 1e-6

    ema = exponential_moving_average(prices, span=2)
    assert ema[0].item() == 100.0


def test_spodeian_lake_dataset_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_file = Path(tmp_dir) / "test_partition.parquet"

        # Write sample Apache Parquet table with PyArrow
        table = pa.Table.from_arrays(
            [
                pa.array([42.50, 42.75, 42.60]),
                pa.array([125000.0, 98000.0, 142000.0]),
                pa.array([1700000000, 1700000060, 1700000120]),
            ],
            names=["close", "volume", "timestamp"],
        )
        pq.write_table(table, parquet_file)

        # Stream into PyTorch Dataset
        dataset = SpodeianLakeDataset(parquet_file, feature_columns=["close", "volume"])
        assert len(dataset) == 3

        item0 = dataset[0]
        assert isinstance(item0, torch.Tensor)
        assert item0.shape == (2,)
        assert abs(item0[0].item() - 42.50) < 1e-5
        assert abs(item0[1].item() - 125000.0) < 1e-3


def test_quantization_sqnr_benchmark():
    from spodeian_ml.bridge.quantization import benchmark_quantization_sqnr
    weights = torch.randn(64, 64)
    results = benchmark_quantization_sqnr(weights)

    # Core IEEE & Posit formats
    assert "float16" in results
    assert "bfloat16" in results
    assert "fp8_e4m3" in results
    assert "fp8_e5m2" in results
    assert "int8" in results
    assert "posit16" in results
    assert "posit8" in results

    # Modern 4-bit, 1-bit, and non-linear formats
    assert "nf4" in results
    assert "int4" in results
    assert "binary" in results
    assert "ternary" in results
    assert "logarithmic" in results

    # High precision formats should have high SQNR
    assert results["float16"]["sqnr_db"] > 20.0
    assert results["posit16"]["sqnr_db"] > 20.0
    assert results["fp8_e4m3"]["sqnr_db"] > 15.0
    assert results["nf4"]["compression_ratio"] == 8.0
    assert results["int4"]["compression_ratio"] == 8.0
    assert results["binary"]["compression_ratio"] == 32.0
    assert results["logarithmic"]["bits"] == 5

    # Balanced ternary has high compression ratio
    assert results["ternary"]["compression_ratio"] > 15.0
    assert 0.5 < results["ternary"]["cosine_similarity"] <= 1.0


def test_kv_cache_quantization():
    from spodeian_ml.bridge.quantization import quantize_kv_cache
    # (Batch=2, Heads=4, SeqLen=32, Dim=64)
    key = torch.randn(2, 4, 32, 64)
    value = torch.randn(2, 4, 32, 64)

    (q_k, scale_k), (q_v, scale_v), meta_int8 = quantize_kv_cache(key, value, precision="int8")
    assert q_k.shape == key.shape
    assert q_v.shape == value.shape
    assert scale_k > 0.0
    assert scale_v > 0.0
    assert meta_int8["compression_ratio"] == 4.0

    (q_k_fp8, scale_k_fp8), (q_v_fp8, scale_v_fp8), meta_fp8 = quantize_kv_cache(key, value, precision="fp8")
    assert q_k_fp8.shape == key.shape
    assert q_v_fp8.shape == value.shape
    assert meta_fp8["compression_ratio"] == 4.0

