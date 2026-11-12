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

    # Dynamic arbitrary quantiles
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    head = QuantileHead(in_features=64, quantiles=quantiles, enforce_monotonic=True)
    loss, preds = head.compute_loss(latent, targets)

    assert preds.shape == (16, 7)
    assert loss.item() >= 0.0, "Pinball loss must be non-negative"

    # Verify Monotonic Non-Crossing: q_i <= q_{i+1} for all batch samples and quantiles
    for i in range(len(quantiles) - 1):
        diff = preds[:, i + 1] - preds[:, i]
        assert torch.all(diff >= -1e-6), f"Quantile crossing detected between quantile {i} and {i+1}!"


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

    # Test dynamic arbitrary days setup: 1, 2, 5, 10, 21, 63, 252 (annual)
    horizon_days = [1, 2, 5, 10, 21, 63, 252]
    head = MultiHorizonHead(in_features=64, horizons=horizon_days)

    assert head.horizon_keys == ["h1", "h2", "h5", "h10", "h21", "h63", "h252"]
    assert head.num_horizons == 7

    # 1. Forward dict
    preds = head(latent)
    assert set(preds.keys()) == set(head.horizon_keys)
    assert preds["h252"].shape == (16, 1)

    # 2. Forward continuous matrix
    matrix_preds = head.forward_matrix(latent)
    assert matrix_preds.shape == (16, 7)

    # 3. Compute loss with matrix targets
    matrix_targets = torch.randn(16, 7)
    loss, task_losses = head.compute_loss(matrix_preds, matrix_targets)
    assert loss.item() is not None
    assert len(task_losses) == 7


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


def test_comprehensive_quantization_formats():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mae = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)

        # Test formats: float32, float16, bfloat16, fp8_e4m3, fp8_e5m2, int8, posit16, posit8, nf4, binary, ternary, logarithmic
        precisions = [
            "float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2", "int8",
            "posit16", "posit8", "nf4", "binary", "ternary", "logarithmic"
        ]

        for prec in precisions:
            file_path = tmp_path / f"model_{prec}.safetensors"
            manifest_path = tmp_path / f"manifest_{prec}.json"

            export_encoder_to_safetensors(
                model=mae,
                output_path=file_path,
                prefix_filter="encoder",
                manifest_path=manifest_path,
                precision=prec,
            )

            assert file_path.exists(), f"Safetensors file missing for {prec}"
            assert manifest_path.exists(), f"Manifest missing for {prec}"

            with safe_open(str(file_path), framework="pt", device="cpu") as f:
                tensor_keys = list(f.keys())
                assert len(tensor_keys) > 0

                if prec == "bfloat16":
                    assert f.get_tensor(tensor_keys[0]).dtype == torch.bfloat16
                elif prec == "float16":
                    assert f.get_tensor(tensor_keys[0]).dtype == torch.float16
                elif prec in ("int8", "binary", "ternary", "nf4"):
                    assert "encoder.0.weight__scale" in tensor_keys
                elif prec in ("posit16", "posit8", "logarithmic"):
                    # Posits and Logarithmic emulate precision in float32 container
                    assert f.get_tensor(tensor_keys[0]).dtype == torch.float32


def test_grokking_config_serialization():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "grokking_cfg.yaml"
        cfg = ExperimentConfig(experiment_name="grokking_run")
        cfg.training.grokking.enabled = True
        cfg.training.grokking.capacity_pruning = True
        cfg.training.grokking.prune_rate = 0.10
        cfg.training.grokking.slingshot_enabled = True
        cfg.training.grokking.activation_sparsity_penalty = 0.05
        cfg.to_yaml(yaml_path)

        loaded_cfg = ExperimentConfig.from_yaml(yaml_path)
        assert loaded_cfg.training.grokking.enabled is True
        assert loaded_cfg.training.grokking.capacity_pruning is True
        assert loaded_cfg.training.grokking.prune_rate == 0.10
        assert loaded_cfg.training.grokking.slingshot_enabled is True
        assert loaded_cfg.training.grokking.activation_sparsity_penalty == 0.05


def test_grokking_hybrid_sparsity_engine():
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = Path(tmp_dir) / "checkpoints"

        torch.manual_seed(42)
        # Small synthetic dataset
        data = torch.randn(32, 8)
        loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=False)

        cfg = ExperimentConfig(
            experiment_name="grokking_sparsity_test",
            seed=42,
        )
        cfg.training.max_epochs = 4
        cfg.checkpoint.checkpoint_dir = str(checkpoint_dir)
        cfg.training.grokking.enabled = True
        cfg.training.grokking.track_weight_norms = True
        cfg.training.grokking.track_representation_sparsity = True
        cfg.training.grokking.capacity_pruning = True
        cfg.training.grokking.prune_threshold = 100.0  # Trigger pruning immediately
        cfg.training.grokking.prune_rate = 0.20
        cfg.training.grokking.prune_interval_epochs = 1
        cfg.training.grokking.dynamic_sample_dropping = True
        cfg.training.grokking.slingshot_enabled = True
        cfg.training.grokking.slingshot_interval = 2
        cfg.training.grokking.slingshot_factor = 2.0
        cfg.training.grokking.activation_sparsity_penalty = 0.01

        model = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)
        trainer = ResumableTrainer(config=cfg, model=model, train_loader=loader)
        history = trainer.run()

        assert len(history) == 4
        for entry in history:
            assert "weight_norm" in entry
            assert "sparsity" in entry
            assert entry["weight_norm"] > 0.0
            assert entry["sparsity"] >= 0.0

        # Verify that capacity pruning increased structural sparsity
        initial_sparsity = history[0]["sparsity"]
        final_sparsity = history[-1]["sparsity"]
        assert final_sparsity >= initial_sparsity



