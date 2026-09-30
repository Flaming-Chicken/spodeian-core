"""
Integration tests for Unified Experiment Registry & Quantization Benchmark Harness.
Verifies LakeRegistry experiment recording, comparison tables, Pareto quantization matrices,
and CLI subcommand interfaces.
"""

import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spodeian_ml.ops.config import ExperimentConfig
from spodeian_ml.ops.trainer import ResumableTrainer
from spodeian_ml.ssl.autoencoder import MaskedAutoencoder
from spodeian_ml.ops.experiment import (
    ExperimentRecord,
    save_experiment_record,
    load_experiment_registry,
    get_experiment_by_id,
    format_experiment_list_table,
    format_experiment_comparison_table,
)
from spodeian_ml.ops.benchmark import (
    run_quantization_benchmark,
    SUPPORTED_BENCHMARK_FORMATS,
)
from spodeian_ml.cli import main


def test_experiment_record_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lake_path = Path(tmp_dir)
        rec = ExperimentRecord(
            run_id="run-test-1234",
            experiment_name="test_experiment",
            config_yaml_snapshot="training:\n  learning_rate: 0.001\n",
            final_metrics={"train_loss": 0.012, "val_loss": 0.015, "best_loss": 0.014},
            hardware_context={"device": "CPU", "python": "3.14"},
            duration_seconds=42.5,
            completion_status="COMPLETED",
            tags=["unit_test", "ssl"],
        )

        reg_file = save_experiment_record(rec, lake_dir=lake_path)
        assert reg_file.exists()

        records = load_experiment_registry(lake_dir=lake_path)
        assert len(records) == 1
        loaded = records[0]
        assert loaded.run_id == "run-test-1234"
        assert loaded.experiment_name == "test_experiment"
        assert loaded.final_metrics["val_loss"] == pytest.approx(0.015)
        assert loaded.tags == ["unit_test", "ssl"]

        # Fetch by ID and prefix
        exact = get_experiment_by_id("run-test-1234", lake_dir=lake_path)
        assert exact is not None
        prefix = get_experiment_by_id("run-test", lake_dir=lake_path)
        assert prefix is not None
        assert prefix.run_id == "run-test-1234"


def test_experiment_comparison_formatting():
    exp_a = ExperimentRecord(
        run_id="run-aaa-1111",
        experiment_name="baseline_model",
        config_yaml_snapshot="lr: 0.001",
        final_metrics={"val_loss": 0.050, "train_loss": 0.045, "best_loss": 0.048},
        hardware_context={"device": "CPU"},
        duration_seconds=100.0,
    )
    exp_b = ExperimentRecord(
        run_id="run-bbb-2222",
        experiment_name="tuned_model",
        config_yaml_snapshot="lr: 0.0005",
        final_metrics={"val_loss": 0.035, "train_loss": 0.030, "best_loss": 0.033},
        hardware_context={"device": "CPU"},
        duration_seconds=120.0,
    )

    diff_table = format_experiment_comparison_table(exp_a, exp_b)
    assert "=== EXPERIMENT COMPARISON ===" in diff_table
    assert "val_loss" in diff_table
    assert "-0.015000" in diff_table  # Delta (0.035 - 0.050)
    assert "+20.00" in diff_table     # Duration delta


def test_trainer_run_emits_experiment_record():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        lake_path = tmp_path / "lake"

        # Monkeypatch default lake path for isolated test
        import spodeian_ml.ops.experiment as exp_mod
        orig_get_default = exp_mod.get_default_lake_path
        exp_mod.get_default_lake_path = lambda: lake_path

        try:
            torch.manual_seed(42)
            x = torch.randn(16, 8)
            loader = DataLoader(TensorDataset(x), batch_size=8, shuffle=False)

            cfg = ExperimentConfig(experiment_name="auto_emit_test")
            cfg.training.max_epochs = 2
            cfg.checkpoint.checkpoint_dir = str(tmp_path / "checkpoints")

            model = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)
            trainer = ResumableTrainer(config=cfg, model=model, train_loader=loader)

            history = trainer.run()
            assert len(history) == 2

            # Verify experiment recorded in lake
            experiments = load_experiment_registry(lake_dir=lake_path)
            assert len(experiments) >= 1
            assert experiments[0].experiment_name == "auto_emit_test"
            assert "final_val_loss" in experiments[0].final_metrics
            assert experiments[0].completion_status == "COMPLETED"
        finally:
            exp_mod.get_default_lake_path = orig_get_default


def test_quantization_benchmark_harness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lake_path = Path(tmp_dir) / "lake"
        model = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)
        cfg = ExperimentConfig(experiment_name="benchmark_test")

        # Monkeypatch default lake path
        import spodeian_ml.ops.experiment as exp_mod
        orig_get_default = exp_mod.get_default_lake_path
        exp_mod.get_default_lake_path = lambda: lake_path

        try:
            results, table_str = run_quantization_benchmark(
                model=model,
                config=cfg,
                eval_samples=64,
                batch_size=16,
                save_to_lake=True,
            )

            # Check that all formats are evaluated
            assert len(results) == len(SUPPORTED_BENCHMARK_FORMATS)
            formats_evaluated = {r["format"] for r in results}
            for fmt_name, _, _ in SUPPORTED_BENCHMARK_FORMATS:
                assert fmt_name in formats_evaluated

            # Table output verification
            assert "UNIVERSAL QUANTIZATION BENCHMARK MATRIX" in table_str
            assert "FP32" in table_str
            assert "INT8" in table_str
            assert "Ternary" in table_str
            assert "BNN" in table_str

            # Verify saved into lake registry
            records = load_experiment_registry(lake_dir=lake_path)
            assert len(records) == 1
            assert "quant_benchmark" in records[0].experiment_name
            assert "fp32_sqnr_db" in records[0].final_metrics
        finally:
            exp_mod.get_default_lake_path = orig_get_default


def test_cli_experiment_and_benchmark(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        lake_dir = Path(tmp_dir) / "lake"
        # Seed two experiments
        exp1 = ExperimentRecord(
            run_id="exp-cli-001",
            experiment_name="run_alpha",
            config_yaml_snapshot="lr: 0.01",
            final_metrics={"val_loss": 0.10, "best_loss": 0.09, "epochs_run": 5},
        )
        exp2 = ExperimentRecord(
            run_id="exp-cli-002",
            experiment_name="run_beta",
            config_yaml_snapshot="lr: 0.005",
            final_metrics={"val_loss": 0.08, "best_loss": 0.07, "epochs_run": 10},
        )
        save_experiment_record(exp1, lake_dir=lake_dir)
        save_experiment_record(exp2, lake_dir=lake_dir)

        # Test CLI: experiment list
        ret = main(["experiment", "list", "--lake-dir", str(lake_dir)])
        assert ret == 0
        out_list = capsys.readouterr().out
        assert "exp-cli-001" in out_list
        assert "exp-cli-002" in out_list

        # Test CLI: experiment compare
        ret = main(["experiment", "compare", "exp-cli-001", "exp-cli-002", "--lake-dir", str(lake_dir)])
        assert ret == 0
        out_compare = capsys.readouterr().out
        assert "=== EXPERIMENT COMPARISON ===" in out_compare
        assert "val_loss" in out_compare

        # Test CLI: benchmark
        ret = main(["benchmark", "--samples", "32", "--batch-size", "16"])
        assert ret == 0
        out_bench = capsys.readouterr().out
        assert "UNIVERSAL QUANTIZATION BENCHMARK MATRIX" in out_bench
