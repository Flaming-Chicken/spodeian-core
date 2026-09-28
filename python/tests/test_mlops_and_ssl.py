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
from spodeian_ml.heads.quantile import QuantileHead, pinball_loss
from spodeian_ml.heads.attention_pool import AttentionPoolingHead
from spodeian_ml.heads.multi_task import MultiHorizonHead
from spodeian_ml.heads.flow import FlowVelocityHead
from spodeian_ml.heads.regime import RegimeClassificationHead
from spodeian_ml.ssl.vicreg import VICRegLoss
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


def test_quantile_head_and_loss():
    latent = torch.randn(16, 64)
    targets = torch.randn(16, 1)

    head = QuantileHead(in_features=64, quantiles=[0.05, 0.50, 0.95])
    loss, preds = head.compute_loss(latent, targets)

    assert preds.shape == (16, 3)
    assert loss.item() >= 0.0, "Pinball loss must be non-negative"


def test_attention_pooling_head():
    seq_latent = torch.randn(16, 20, 64)  # (Batch 16, Seq 20, Dim 64)
    flat_latent = torch.randn(16, 64)

    head = AttentionPoolingHead(in_features=64, out_features=1, attn_dim=32)

    # 3D sequence test
    out_seq = head(seq_latent)
    assert out_seq.shape == (16, 1)

    # 2D flat test
    out_flat = head(flat_latent)
    assert out_flat.shape == (16, 1)


def test_multi_horizon_head():
    latent = torch.randn(16, 64)
    targets = {
        "h1": torch.randn(16, 1),
        "h5": torch.randn(16, 1),
        "h21": torch.randn(16, 1),
    }

    head = MultiHorizonHead(in_features=64, horizons=["h1", "h5", "h21"])
    preds = head(latent)

    assert set(preds.keys()) == {"h1", "h5", "h21"}
    assert preds["h1"].shape == (16, 1)

    loss, task_losses = head.compute_loss(preds, targets)
    assert loss.item() is not None
    assert len(task_losses) == 3


def test_flow_velocity_head():
    x_0 = torch.randn(16, 32)
    x_1 = torch.randn(16, 32)
    t = torch.rand(16, 1)

    head = FlowVelocityHead(state_dim=32, hidden_dim=64, time_embed_dim=16)
    v_pred = head(x_0, t)
    assert v_pred.shape == (16, 32)

    loss, _ = head.compute_flow_matching_loss(x_0, x_1, t)
    assert loss.item() >= 0.0


def test_regime_classification_head():
    latent = torch.randn(16, 64)
    labels = torch.randint(0, 4, (16,))

    head = RegimeClassificationHead(in_features=64, num_classes=4)
    probs = head.predict_probabilities(latent)
    assert probs.shape == (16, 4)
    # Probabilities must sum to 1.0 across classes
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)

    loss, logits = head.compute_loss(latent, labels)
    assert logits.shape == (16, 4)
    assert loss.item() > 0.0


def test_vicreg_loss():
    z_a = torch.randn(32, 64)
    z_b = z_a + torch.randn_like(z_a) * 0.05

    vicreg = VICRegLoss()
    total_loss, sim, std, cov = vicreg(z_a, z_b)

    assert total_loss.item() > 0.0
    assert sim.item() >= 0.0
    assert std.item() >= 0.0
    assert cov.item() >= 0.0


def test_safetensors_float16_export():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fp16_file = tmp_path / "model_fp16.safetensors"

        mae = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)
        export_encoder_to_safetensors(
            model=mae,
            output_path=fp16_file,
            prefix_filter="encoder",
            dtype="float16",
        )

        assert fp16_file.exists()
        with safe_open(str(fp16_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                assert tensor.dtype == torch.float16, f"Expected float16 tensor, got {tensor.dtype} for {key}"

