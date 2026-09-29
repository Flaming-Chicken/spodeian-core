"""
Unified CLI for spodeian-ml: Training, MLOps, Model Verification, and Interactive TUI Studio.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from spodeian_ml.ops.config import ExperimentConfig
from spodeian_ml.ops.trainer import ResumableTrainer
from spodeian_ml.ops.tui import run_interactive_training
from spodeian_ml.ssl.autoencoder import MaskedAutoencoder


def build_synthetic_dataloader(dim: int = 16, samples: int = 256, batch_size: int = 32) -> DataLoader:
    """Creates a deterministic synthetic dataloader for interactive demonstration."""
    torch.manual_seed(42)
    x = torch.randn(samples, dim)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)


def handle_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        sys.stderr.write(f"[-] Error: Config file not found: {config_path}\n")
        return 1

    config = ExperimentConfig.from_yaml(config_path)
    if args.epochs:
        config.training.max_epochs = args.epochs

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Instantiate model
    model = MaskedAutoencoder(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
        mask_ratio=config.ssl.mask_ratio,
    )

    # For CLI demo, construct or load data
    train_loader = build_synthetic_dataloader(dim=config.model.input_dim, batch_size=config.training.batch_size)
    val_loader = build_synthetic_dataloader(dim=config.model.input_dim, samples=64, batch_size=config.training.batch_size)

    trainer = ResumableTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    if args.tui:
        run_interactive_training(trainer, max_epochs=config.training.max_epochs)
    else:
        history = trainer.run()
        print(f"[+] Completed {len(history)} epochs. Final val loss: {history[-1]['val_loss']:.5f}")

    return 0


def handle_studio(args: argparse.Namespace) -> int:
    """Launch the interactive Flight Deck Studio immediately."""
    config_path = Path(args.config) if args.config else Path("configs/risky_business_ssl_v1.yaml")
    if config_path.exists():
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig(experiment_name="studio_session")
        config.training.grokking.enabled = True
        config.training.grokking.capacity_pruning = True
        config.training.grokking.slingshot_enabled = True

    if args.epochs:
        config.training.max_epochs = args.epochs

    device = torch.device("cpu")
    model = MaskedAutoencoder(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
    )
    train_loader = build_synthetic_dataloader(dim=config.model.input_dim, batch_size=config.training.batch_size)
    val_loader = build_synthetic_dataloader(dim=config.model.input_dim, samples=64, batch_size=config.training.batch_size)

    trainer = ResumableTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    run_interactive_training(trainer, max_epochs=config.training.max_epochs)
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spodeian-ml",
        description="Spodeian ML: Foundation Training, MLOps, and Autonomous Flight Deck Studio",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # train command
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML experiment config")
    train_parser.add_argument("--tui", action="store_true", help="Launch interactive 4-quadrant TUI flight deck")
    train_parser.add_argument("--epochs", "-e", type=int, default=None, help="Override max epochs")
    train_parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")

    # studio command
    studio_parser = subparsers.add_parser("studio", help="Launch interactive TUI flight deck studio directly")
    studio_parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML experiment config")
    studio_parser.add_argument("--epochs", "-e", type=int, default=50, help="Max epochs")

    args = parser.parse_args(argv)

    if args.command == "train":
        return handle_train(args)
    elif args.command == "studio":
        return handle_studio(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
