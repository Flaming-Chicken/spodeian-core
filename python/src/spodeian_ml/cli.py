"""
Unified CLI for spodeian-ml: Training, MLOps, Model Verification, and Interactive TUI Studio.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from spodeian_ml.data.lake_loader import (
    build_synthetic_dataloader,
    build_train_val_dataloaders,
    SpodeianLakeDataset,
)
from spodeian_ml.ops.config import ExperimentConfig
from spodeian_ml.ops.trainer import ResumableTrainer
from spodeian_ml.ops.tui import run_interactive_training
from spodeian_ml.ssl.autoencoder import MaskedAutoencoder


def handle_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        sys.stderr.write(f"[-] Error: Config file not found: {config_path}\n")
        return 1

    config = ExperimentConfig.from_yaml(config_path)
    if args.epochs:
        config.training.max_epochs = args.epochs

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_path = getattr(args, "data_path", None)
    lake_dataset = getattr(args, "lake_dataset", None)
    lake_dir = getattr(args, "lake_dir", None)

    train_loader, val_loader, input_dim = build_train_val_dataloaders(
        data_path=data_path,
        lake_dataset=lake_dataset,
        lake_dir=lake_dir,
        batch_size=config.training.batch_size,
        dim=config.model.input_dim,
    )

    if input_dim != config.model.input_dim:
        config.model.input_dim = input_dim

    # Instantiate model
    model = MaskedAutoencoder(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
        mask_ratio=config.ssl.mask_ratio,
    )

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

    data_path = getattr(args, "data_path", None)
    lake_dataset = getattr(args, "lake_dataset", None)
    lake_dir = getattr(args, "lake_dir", None)

    train_loader, val_loader, input_dim = build_train_val_dataloaders(
        data_path=data_path,
        lake_dataset=lake_dataset,
        lake_dir=lake_dir,
        batch_size=config.training.batch_size,
        dim=config.model.input_dim,
    )

    if input_dim != config.model.input_dim:
        config.model.input_dim = input_dim

    model = MaskedAutoencoder(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
    )

    trainer = ResumableTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )


    history = run_interactive_training(
        trainer,
        max_epochs=config.training.max_epochs,
        headless=args.headless,
        resume=args.resume,
    )
    if args.headless:
        import json

        summary = {
            "total_epochs": len(history),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "best_loss": trainer.best_loss,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "weight_decay": trainer.optimizer.param_groups[0]["weight_decay"],
            "global_step": trainer.global_step,
        }
        print(json.dumps(summary, indent=2))
    return 0


def handle_experiment(args: argparse.Namespace) -> int:
    from spodeian_ml.ops.experiment import (
        load_experiment_registry,
        get_experiment_by_id,
        format_experiment_list_table,
        format_experiment_comparison_table,
    )
    lake_dir = Path(args.lake_dir) if getattr(args, "lake_dir", None) else None

    if args.exp_action == "list":
        experiments = load_experiment_registry(lake_dir)
        print(format_experiment_list_table(experiments))
        return 0
    elif args.exp_action == "compare":
        exp_a = get_experiment_by_id(args.run_id_a, lake_dir)
        exp_b = get_experiment_by_id(args.run_id_b, lake_dir)
        if not exp_a:
            sys.stderr.write(f"[-] Error: Experiment run_id not found: {args.run_id_a}\n")
            return 1
        if not exp_b:
            sys.stderr.write(f"[-] Error: Experiment run_id not found: {args.run_id_b}\n")
            return 1
        print(format_experiment_comparison_table(exp_a, exp_b))
        return 0
    else:
        sys.stderr.write("[-] Error: Use 'spodeian-ml experiment list' or 'spodeian-ml experiment compare <id_a> <id_b>'\n")
        return 1


def handle_benchmark(args: argparse.Namespace) -> int:
    from spodeian_ml.ops.benchmark import run_quantization_benchmark
    config_path = Path(args.config) if args.config else Path("configs/risky_business_ssl_v1.yaml")
    if config_path.exists():
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig(experiment_name="benchmark_run")

    model = MaskedAutoencoder(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
    )

    if args.model:
        m_path = Path(args.model)
        if m_path.exists():
            if m_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(m_path))
                model.load_state_dict(state, strict=False)
            elif m_path.suffix == ".pt":
                state = torch.load(m_path, map_location="cpu")
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(state, strict=False)

    _, table_str = run_quantization_benchmark(
        model=model,
        config=config,
        eval_samples=args.samples,
        batch_size=args.batch_size,
    )
    print(table_str)
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
    train_parser.add_argument("--data-path", type=str, default=None, help="Path to Parquet file or directory of partitions")
    train_parser.add_argument("--lake-dataset", type=str, default=None, help="Dataset ID to load from Spodeian Lake registry")
    train_parser.add_argument("--lake-dir", type=str, default=None, help="Path to custom Spodeian Lake directory")
    train_parser.add_argument("--tui", action="store_true", help="Launch interactive 4-quadrant TUI flight deck")
    train_parser.add_argument("--epochs", "-e", type=int, default=None, help="Override max epochs")
    train_parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")

    # studio command
    studio_parser = subparsers.add_parser("studio", help="Launch interactive TUI flight deck studio directly")
    studio_parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML experiment config")
    studio_parser.add_argument("--data-path", type=str, default=None, help="Path to Parquet file or directory of partitions")
    studio_parser.add_argument("--lake-dataset", type=str, default=None, help="Dataset ID to load from Spodeian Lake registry")
    studio_parser.add_argument("--lake-dir", type=str, default=None, help="Path to custom Spodeian Lake directory")
    studio_parser.add_argument("--epochs", "-e", type=int, default=50, help="Max epochs")
    studio_parser.add_argument("--headless", action="store_true", help="Run headlessly without interactive terminal rendering")
    studio_parser.add_argument("--resume", action="store_true", help="Resume tuning parameters from studio_session.json")


    # experiment command
    experiment_parser = subparsers.add_parser("experiment", help="Query and compare Spodeian Lake experiment runs")
    exp_subparsers = experiment_parser.add_subparsers(dest="exp_action", help="Experiment sub-action")
    exp_list = exp_subparsers.add_parser("list", help="List all recorded experiments")
    exp_list.add_argument("--lake-dir", type=str, default=None, help="Path to custom lake directory")
    exp_compare = exp_subparsers.add_parser("compare", help="Compare two experiment runs side-by-side")
    exp_compare.add_argument("run_id_a", type=str, help="First run ID or prefix")
    exp_compare.add_argument("run_id_b", type=str, help="Second run ID or prefix")
    exp_compare.add_argument("--lake-dir", type=str, default=None, help="Path to custom lake directory")

    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run Universal Quantization Benchmark Matrix")
    benchmark_parser.add_argument("--model", "-m", type=str, default=None, help="Path to model weights (.safetensors or .pt)")
    benchmark_parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML experiment config")
    benchmark_parser.add_argument("--samples", type=int, default=512, help="Number of benchmark evaluation samples")
    benchmark_parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")

    args = parser.parse_args(argv)

    if args.command == "train":
        return handle_train(args)
    elif args.command == "studio":
        return handle_studio(args)
    elif args.command == "experiment":
        return handle_experiment(args)
    elif args.command == "benchmark":
        return handle_benchmark(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
