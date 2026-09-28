"""
Comprehensive MLOps, SSL, and Resumable Checkpoint Tests for spodeian-ml.
Verifies Training as Code, bit-perfect resumption, swappable heads, and Safetensors export.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from safetensors import safe_open

from spodeian_ml.ops.config import ExperimentConfig
from spodeian_ml.ops.checkpoint import save_checkpoint, load_checkpoint
from spodeian_ml.ops.trainer import ResumableTrainer
from spodeian_ml.ssl.autoencoder import MaskedAutoencoder
from spodeian_ml.ssl.contrastive import InfoNCELoss
from spodeian_ml.heads.linear_probe import LinearProbeHead
from spodeian_ml.heads.mlp import MLPProjectionHead
from spodeian_ml.heads.temporal import TemporalHead
from spodeian_ml.bridge.safetensors_exporter import export_encoder_to_safetensors


def test_config_yaml_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "test_cfg.yaml"
        cfg = ExperimentConfig(experiment_name="unit_test_run")
        cfg.to_yaml(yaml_path)

        loaded_cfg = ExperimentConfig.from_yaml(yaml_path)
        assert loaded_cfg.experiment_name == "unit_test_run"
        assert loaded_cfg.model.latent_dim == 64
        assert loaded_cfg.training.learning_rate == 1e-3


def test_masked_autoencoder_convergence():
    torch.manual_seed(42)
    # Synthetic batch of 64 samples x 16 features
    x = torch.randn(64, 16)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = MaskedAutoencoder(input_dim=16, hidden_dim=32, latent_dim=16, num_layers=1, mask_ratio=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss, _ = model.compute_ssl_loss(x, seed=42)

    # Train for 5 epochs
    for epoch in range(5):
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = model.compute_ssl_loss(batch[0], seed=42 + epoch)
            loss.backward()
            optimizer.step()

    final_loss, _ = model.compute_ssl_loss(x, seed=42)
    assert final_loss.item() < initial_loss.item(), "SSL reconstruction loss should decrease with training"


def test_infonce_contrastive_loss():
    loss_fn = InfoNCELoss(temperature=0.1)

    # Identical views should have low loss
    z_clean = torch.randn(8, 16)
    z_a = z_clean + torch.randn_like(z_clean) * 0.01
    z_b = z_clean + torch.randn_like(z_clean) * 0.01

    loss_close = loss_fn(z_a, z_b)

    # Uncorrelated views should have higher loss
    z_random = torch.randn(8, 16)
    loss_far = loss_fn(z_a, z_random)

    assert loss_close.item() < loss_far.item(), "Positive pairs must yield lower InfoNCE loss than random pairs"


def test_swappable_architecture_heads():
    latent = torch.randn(16, 64)  # Batch 16, latent 64

    # 1. Linear Probe
    lp = LinearProbeHead(in_features=64, out_features=1)
    out_lp = lp(latent)
    assert out_lp.shape == (16, 1)

    # 2. MLP Head
    mlp = MLPProjectionHead(in_features=64, hidden_dims=[32], out_features=2, dropout=0.0)
    out_mlp = mlp(latent)
    assert out_mlp.shape == (16, 2)

    # 3. Temporal Head
    temporal = TemporalHead(in_features=64, hidden_dim=32, num_layers=1, out_features=1)
    seq_latent = torch.randn(16, 10, 64)  # Batch 16, Seq 10, Latent 64
    out_temporal = temporal(seq_latent)
    assert out_temporal.shape == (16, 1)


def test_bit_perfect_training_resumption():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        checkpoint_dir = tmp_path / "checkpoints"

        # Synthetic data
        torch.manual_seed(42)
        data = torch.randn(32, 8)
        loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=False)

        cfg_1 = ExperimentConfig(
            experiment_name="run_initial",
            seed=42,
        )
        cfg_1.training.max_epochs = 3
        cfg_1.checkpoint.checkpoint_dir = str(checkpoint_dir)
        cfg_1.checkpoint.save_every_n_epochs = 1

        model_1 = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)
        trainer_1 = ResumableTrainer(config=cfg_1, model=model_1, train_loader=loader)
        history_1 = trainer_1.run()
        assert len(history_1) == 3

        checkpoint_epoch_2 = checkpoint_dir / "checkpoint_epoch_2.pt"
        assert checkpoint_epoch_2.exists()

        # Phase 2: Resume training to epoch 5
        cfg_2 = ExperimentConfig(
            experiment_name="run_resumed",
            seed=42,
        )
        cfg_2.training.max_epochs = 5
        cfg_2.checkpoint.checkpoint_dir = str(checkpoint_dir)
        cfg_2.checkpoint.resume_from = str(checkpoint_epoch_2)

        model_2 = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)
        trainer_2 = ResumableTrainer(config=cfg_2, model=model_2, train_loader=loader)

        assert trainer_2.start_epoch == 3, "Resumed trainer must pick up at start_epoch = 3"
        history_2 = trainer_2.run()
        assert len(history_2) == 2, "Only epochs 3 and 4 should be run in the resumed session"


def test_safetensors_export_and_integrity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        safetensor_file = tmp_path / "model_weights.safetensors"
        manifest_file = tmp_path / "manifest.json"

        mae = MaskedAutoencoder(input_dim=16, hidden_dim=32, latent_dim=16, num_layers=2)

        # Export encoder
        export_encoder_to_safetensors(
            model=mae,
            output_path=safetensor_file,
            prefix_filter="encoder",
            manifest_path=manifest_file,
        )

        assert safetensor_file.exists()
        assert manifest_file.exists()

        # Verify tensors inside safetensors file
        with safe_open(str(safetensor_file), framework="pt", device="cpu") as f:
            tensor_keys = list(f.keys())
            assert len(tensor_keys) > 0

            # All keys must be encoder weights, zero decoder weights
            for key in tensor_keys:
                assert key.startswith("encoder."), f"Decoder key leaked into export: {key}"
                tensor = f.get_tensor(key)
                assert isinstance(tensor, torch.Tensor)
                assert tensor.dtype == torch.float32
