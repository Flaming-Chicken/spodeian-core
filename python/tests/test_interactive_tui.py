"""
Integration tests for spodeian-ml Autonomous & Tweakable Training TUI.
Verifies flight deck telemetry, 4-quadrant layout, non-blocking key dispatcher,
granular tuning drawer, and hotkey actions.
"""

import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spodeian_ml.ops.config import ExperimentConfig
from spodeian_ml.ops.trainer import ResumableTrainer
from spodeian_ml.ops.tui import (
    InteractiveTrainingStudio,
    NonBlockingInputReader,
    GranularTuningDrawer,
    run_interactive_training,
    format_sparkline,
)
from spodeian_ml.ssl.autoencoder import MaskedAutoencoder


def create_test_trainer(tmp_dir: Path) -> ResumableTrainer:
    torch.manual_seed(42)
    x = torch.randn(32, 8)
    loader = DataLoader(TensorDataset(x), batch_size=8, shuffle=False)

    cfg = ExperimentConfig(experiment_name="tui_test_run", seed=42)
    cfg.training.max_epochs = 10
    cfg.training.learning_rate = 0.001
    cfg.training.weight_decay = 0.0001
    cfg.training.grokking.enabled = True
    cfg.training.grokking.capacity_pruning = True
    cfg.training.grokking.slingshot_enabled = True
    cfg.checkpoint.checkpoint_dir = str(tmp_dir / "checkpoints")

    model = MaskedAutoencoder(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=1)
    trainer = ResumableTrainer(config=cfg, model=model, train_loader=loader, val_loader=loader)
    return trainer


def test_sparkline_generator():
    vals = [0.1, 0.2, 0.5, 0.8, 1.0]
    spark = format_sparkline(vals, length=5)
    assert len(spark) == 5
    assert spark[0] == " "
    assert spark[-1] == "█"

    empty_spark = format_sparkline([], length=8)
    assert len(empty_spark) == 8


def test_tui_initial_state_and_stepping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        reader = NonBlockingInputReader()
        studio = InteractiveTrainingStudio(trainer, input_reader=reader)

        assert not studio.is_paused
        assert studio.is_running
        assert not studio.drawer.is_open
        assert studio.current_epoch == 0

        # Run 2 steps
        entry1 = studio.step()
        assert entry1 is not None
        assert studio.current_epoch == 1
        assert len(studio.history_train_loss) == 1
        assert len(studio.history_val_loss) == 1
        assert len(studio.history_weight_norm) == 1

        entry2 = studio.step()
        assert entry2 is not None
        assert studio.current_epoch == 2
        assert len(studio.history_train_loss) == 2


def test_tui_hotkeys_pause_and_lr():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        reader = NonBlockingInputReader()
        studio = InteractiveTrainingStudio(trainer, input_reader=reader)

        # Pause toggle via Space
        reader.push_key(" ")
        studio.step()
        assert studio.is_paused
        # While paused, step returns None and epoch doesn't advance
        epoch_before = studio.current_epoch
        res = studio.step()
        assert res is None
        assert studio.current_epoch == epoch_before

        # Unpause via Space
        reader.push_key(" ")
        studio.step()
        assert not studio.is_paused

        # Nudge LR UP via '+'
        old_lr = trainer.optimizer.param_groups[0]["lr"]
        reader.push_key("+")
        studio.step()
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        assert new_lr > old_lr

        # Nudge LR DOWN via '-'
        reader.push_key("-")
        studio.step()
        reverted_lr = trainer.optimizer.param_groups[0]["lr"]
        assert reverted_lr < new_lr


def test_tui_hotkeys_slingshot_prune_and_swa():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        reader = NonBlockingInputReader()
        studio = InteractiveTrainingStudio(trainer, input_reader=reader)

        # Slingshot kick 'k'
        init_lr = trainer.optimizer.param_groups[0]["lr"]
        reader.push_key("k")
        studio.step()
        assert trainer.optimizer.param_groups[0]["lr"] > init_lr
        assert any("Slingshot Kick applied" in ev for ev in studio.events)

        # Capacity pruning 'c'
        reader.push_key("c")
        studio.step()
        assert any("Capacity Pruned" in ev for ev in studio.events)

        # SWA settle 'w'
        reader.push_key("w")
        studio.step()
        assert any("SWA weights updated" in ev for ev in studio.events)

        # KV Cache eviction toggle 'e'
        old_mode = studio.kv_eviction_mode
        reader.push_key("e")
        studio.step()
        assert studio.kv_eviction_mode != old_mode

        # Checkpoint export 's'
        reader.push_key("s")
        studio.step()
        assert any("Forced manual checkpoint exported" in ev for ev in studio.events)


def test_tui_granular_tuning_drawer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        reader = NonBlockingInputReader()
        studio = InteractiveTrainingStudio(trainer, input_reader=reader)

        assert not studio.drawer.is_open

        # Open drawer with 'g'
        reader.push_key("g")
        studio.handle_input()
        assert studio.drawer.is_open

        # Navigate down 'j'
        sel_idx = studio.drawer.selected_idx
        reader.push_key("j")
        studio.handle_input()
        assert studio.drawer.selected_idx == (sel_idx + 1) % len(studio.drawer.params)

        # Navigate up 'k'
        reader.push_key("k")
        studio.handle_input()
        assert studio.drawer.selected_idx == sel_idx

        # Adjust right '+'
        p = studio.drawer.params[studio.drawer.selected_idx]
        init_v = p.value
        reader.push_key("+")
        studio.handle_input()
        assert p.value > init_v

        # Adjust left '-'
        reader.push_key("-")
        studio.handle_input()
        assert p.value <= init_v * 1.0001

        # Close drawer with 'g'
        reader.push_key("g")
        studio.handle_input()
        assert not studio.drawer.is_open


def test_tui_rendering_output():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        reader = NonBlockingInputReader()
        studio = InteractiveTrainingStudio(trainer, input_reader=reader)
        studio.step()

        # Closed drawer rendering
        frame = studio.render(clear_screen=False)
        assert "SPODEIAN-ML · Studio ►" in frame
        assert "Train Loss:" in frame
        assert "L2 Weight Norm:" in frame
        assert "KV Cache:" in frame
        assert "GRANULAR TUNING DRAWER" not in frame

        # Open drawer rendering
        studio.drawer.is_open = True
        frame_drawer = studio.render(clear_screen=False)
        assert "GRANULAR TUNING DRAWER [g]" in frame_drawer
        assert "Learning Rate" in frame_drawer


def test_tui_help_overlay_toggle_and_content():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        reader = NonBlockingInputReader()
        studio = InteractiveTrainingStudio(trainer, input_reader=reader)

        assert not studio.show_help_overlay

        # Toggle help with '?'
        reader.push_key("?")
        studio.handle_input()
        assert studio.show_help_overlay

        # Render help overlay
        help_frame = studio.render(clear_screen=False)
        assert "HELP & TELEMETRY OVERLAY" in help_frame
        assert "COMMAND SHORTCUTS" in help_frame
        assert "[Space] / [p]" in help_frame
        assert "[g]" in help_frame
        assert "ACTIVE ENGINE PARAMETERS & TELEMETRY" in help_frame
        assert "Active LR:" in help_frame
        assert "KV Eviction Mode:" in help_frame

        # Dismiss with 'h'
        reader.push_key("h")
        studio.handle_input()
        assert not studio.show_help_overlay

        # Toggle again and dismiss with Esc
        reader.push_key("h")
        studio.handle_input()
        assert studio.show_help_overlay
        reader.push_key("\x1b")
        studio.handle_input()
        assert not studio.show_help_overlay


def test_tui_curriculum_phase_detection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        studio = InteractiveTrainingStudio(trainer)

        assert studio.get_curriculum_phase() == "Init"

        # Simulate early high loss
        studio.history_train_loss.append(1.5)
        studio.history_val_loss.append(1.6)
        assert "Memorization" in studio.get_curriculum_phase()

        # Simulate low train loss with overfit counter
        studio.history_train_loss.append(0.01)
        studio.history_val_loss.append(0.5)
        studio.overfit_counter = 3
        assert "Grokking Transition" in studio.get_curriculum_phase()

        # Simulate converged generalization (below prune_threshold of 0.001)
        studio.history_train_loss.append(0.0005)
        studio.history_val_loss.append(0.0006)
        studio.overfit_counter = 0
        assert "Generalization" in studio.get_curriculum_phase()


def test_tui_session_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        session_path = Path(tmp_dir) / "studio_session.json"
        trainer = create_test_trainer(Path(tmp_dir))
        studio = InteractiveTrainingStudio(trainer)

        # Modify values
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = 0.005
            pg["weight_decay"] = 0.02
        studio.kv_eviction_mode = "AttentionSoftmax"
        studio.last_checkpoint_path = str(Path(tmp_dir) / "ckpt.pt")

        # Save session
        studio.save_session_state(session_path)
        assert session_path.exists()

        # Create fresh studio and load
        trainer2 = create_test_trainer(Path(tmp_dir))
        studio2 = InteractiveTrainingStudio(trainer2)
        assert studio2.trainer.optimizer.param_groups[0]["lr"] != 0.005

        loaded = studio2.load_session_state(session_path)
        assert loaded
        assert studio2.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
        assert studio2.trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)
        assert studio2.kv_eviction_mode == "AttentionSoftmax"
        assert studio2.last_checkpoint_path == str(Path(tmp_dir) / "ckpt.pt")


def test_run_interactive_training_loop():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = create_test_trainer(Path(tmp_dir))
        reader = NonBlockingInputReader()
        # Enqueue 3 steps and then 'q' to quit
        reader.push_key("k")
        reader.push_key("q")

        history = run_interactive_training(
            trainer,
            max_epochs=4,
            headless=True,
            input_reader=reader,
            render_interval=0.001,
        )
        assert len(history) >= 1


def test_cli_studio_headless(capsys):
    from spodeian_ml.cli import main
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "test_cfg.yaml"
        cfg = ExperimentConfig(experiment_name="cli_headless_test")
        cfg.training.max_epochs = 2
        cfg.to_yaml(cfg_file)

        ret = main(["studio", "--config", str(cfg_file), "--epochs", "2", "--headless"])
        assert ret == 0
        captured = capsys.readouterr()
        import json
        summary = json.loads(captured.out)
        assert summary["total_epochs"] == 2
        assert "learning_rate" in summary
