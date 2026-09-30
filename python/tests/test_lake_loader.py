"""
Tests for Spodeian Lake Dataset, Parquet streaming, Registry resolution, and DataLoader factory.
"""

import json
import hashlib
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from spodeian_ml.data.lake_loader import (
    SpodeianLakeDataset,
    build_dataloader,
    build_train_val_dataloaders,
    load_lake_registry,
)
from spodeian_ml.cli import main


@pytest.fixture
def temp_lake(tmp_path: Path):
    """Sets up a temporary data lake directory with Parquet partitions and a registry."""
    lake_dir = tmp_path / "lake"
    data_dir = lake_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create Partition 1 (earlier timestamp)
    schema = pa.schema([
        ("feature_a", pa.float32()),
        ("feature_b", pa.float32()),
        ("target", pa.float32()),
    ])
    data_p1 = {
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0],
        "target": [0.1, 0.2, 0.3, 0.4],
    }
    table_p1 = pa.Table.from_pydict(data_p1, schema=schema)
    path_p1 = data_dir / "ticks_2026_01.parquet"
    pq.write_table(table_p1, path_p1)
    sha_p1 = hashlib.sha256(path_p1.read_bytes()).hexdigest()

    # Create Partition 2 (later timestamp)
    data_p2 = {
        "feature_a": [5.0, 6.0, 7.0, 8.0],
        "feature_b": [50.0, 60.0, 70.0, 80.0],
        "target": [0.5, 0.6, 0.7, 0.8],
    }
    table_p2 = pa.Table.from_pydict(data_p2, schema=schema)
    path_p2 = data_dir / "ticks_2026_02.parquet"
    pq.write_table(table_p2, path_p2)
    sha_p2 = hashlib.sha256(path_p2.read_bytes()).hexdigest()

    # Create Registry JSON
    registry_data = {
        "partitions": {
            "financial_ticks:2026_01": {
                "dataset_id": "financial_ticks",
                "partition_key": "2026_01",
                "file_path": str(path_p1),
                "record_count": 4,
                "byte_size": path_p1.stat().st_size,
                "sha256_checksum": sha_p1,
                "created_at_epoch_s": 1000,
                "last_accessed_epoch_s": 1000,
                "ttl_seconds": None,
                "training_utility": 0.5,
                "anomaly_score": 0.1,
                "training_epochs_seen": 0,
                "is_pinned": False,
            },
            "financial_ticks:2026_02": {
                "dataset_id": "financial_ticks",
                "partition_key": "2026_02",
                "file_path": str(path_p2),
                "record_count": 4,
                "byte_size": path_p2.stat().st_size,
                "sha256_checksum": sha_p2,
                "created_at_epoch_s": 2000,
                "last_accessed_epoch_s": 2000,
                "ttl_seconds": None,
                "training_utility": 0.5,
                "anomaly_score": 0.1,
                "training_epochs_seen": 0,
                "is_pinned": False,
            },
        },
        "experiments": {},
    }
    reg_file = lake_dir / "lake_registry.json"
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(registry_data, f)

    return lake_dir, path_p1, path_p2, data_dir


def test_spodeian_lake_dataset_single_file(temp_lake):
    _, path_p1, _, _ = temp_lake
    ds = SpodeianLakeDataset(path_p1)
    assert len(ds) == 4
    assert ds.features.shape == (4, 3)
    sample = ds[0]
    assert sample.shape == (3,)
    assert torch.isclose(sample[0], torch.tensor(1.0))
    assert torch.isclose(sample[1], torch.tensor(10.0))


def test_spodeian_lake_dataset_directory(temp_lake):
    _, _, _, data_dir = temp_lake
    ds = SpodeianLakeDataset(data_dir)
    # 4 rows from p1 + 4 rows from p2 = 8 rows
    assert len(ds) == 8
    assert ds.features.shape == (8, 3)


def test_spodeian_lake_dataset_from_registry(temp_lake):
    lake_dir, _, _, _ = temp_lake
    ds = SpodeianLakeDataset.from_registry(
        dataset_id="financial_ticks",
        lake_dir=lake_dir,
        verify_sha256=True,
    )
    assert len(ds) == 8
    # First row should be from partition 1 (1000 epoch < 2000 epoch)
    assert torch.isclose(ds[0][0], torch.tensor(1.0))
    # Fifth row should be from partition 2
    assert torch.isclose(ds[4][0], torch.tensor(5.0))


def test_sha256_tamper_detection(temp_lake):
    lake_dir, path_p1, _, _ = temp_lake
    # Tamper with file p1
    with open(path_p1, "ab") as f:
        f.write(b"corrupted_bytes")

    with pytest.raises(ValueError, match="Integrity check failed"):
        SpodeianLakeDataset.from_registry(
            dataset_id="financial_ticks",
            lake_dir=lake_dir,
            verify_sha256=True,
        )


def test_build_dataloader_factory(temp_lake):
    lake_dir, path_p1, _, _ = temp_lake
    # Synthetic fallback
    synthetic_loader = build_dataloader(dim=8, samples=32, batch_size=4)
    batch = next(iter(synthetic_loader))
    assert batch[0].shape == (4, 8)

    # From file path
    file_loader = build_dataloader(data_path=path_p1, batch_size=2)
    batch = next(iter(file_loader))
    assert batch.shape == (2, 3)

    # From lake dataset
    lake_loader = build_dataloader(lake_dataset="financial_ticks", lake_dir=lake_dir, batch_size=4)
    batch = next(iter(lake_loader))
    assert batch.shape == (4, 3)


def test_build_train_val_dataloaders(temp_lake):
    lake_dir, _, _, _ = temp_lake
    train_loader, val_loader, dim = build_train_val_dataloaders(
        lake_dataset="financial_ticks",
        lake_dir=lake_dir,
        batch_size=2,
        val_ratio=0.25,
    )
    assert dim == 3
    # Total samples: 8 -> train: 6, val: 2
    train_samples = sum(len(b) for b in train_loader)
    val_samples = sum(len(b) for b in val_loader)
    assert train_samples == 6
    assert val_samples == 2


def test_cli_train_with_lake_dataset(temp_lake, tmp_path):
    lake_dir, _, _, _ = temp_lake
    # Create simple config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
experiment_name: "cli_lake_test"
model:
  input_dim: 3
  hidden_dim: 8
  latent_dim: 4
  num_layers: 1
training:
  max_epochs: 2
  batch_size: 2
  learning_rate: 0.001
""")

    code = main([
        "train",
        "--config", str(config_file),
        "--lake-dataset", "financial_ticks",
        "--lake-dir", str(lake_dir),
        "--epochs", "2",
    ])
    assert code == 0


def test_cli_studio_headless_with_data_path(temp_lake):
    _, path_p1, _, _ = temp_lake
    code = main([
        "studio",
        "--data-path", str(path_p1),
        "--headless",
        "--epochs", "2",
    ])
    assert code == 0
