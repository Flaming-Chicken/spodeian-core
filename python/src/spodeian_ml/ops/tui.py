"""
Autonomous & Tweakable Training TUI for spodeian-ml.
Provides real-time interactive steering, grokking & sparsity flight deck,
4-quadrant MLOps telemetry, and non-blocking granular tuning drawer.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import math

import torch
import torch.nn as nn

from spodeian_ml.ops.trainer import ResumableTrainer
from spodeian_ml.ops.checkpoint import save_checkpoint
from spodeian_ml.ops.paged_attention import PagedKVCacheManager
from spodeian_ml.ops.smoothing import ModelWeightAverager


# ANSI Color Codes
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"
COLOR_BLUE = "\033[34m"
COLOR_WHITE = "\033[37m"
COLOR_BG_DRAWER = "\033[44m"


def format_sparkline(values: List[float], length: int = 18) -> str:
    """Generates a Unicode sparkline string for a list of numerical values."""
    if not values:
        return " " * length
    sample = values[-length:] if len(values) >= length else values
    min_v = min(sample)
    max_v = max(sample)
    bars = " ▂▃▄▅▆▇█"
    
    if abs(max_v - min_v) < 1e-9:
        line = bars[3] * len(sample)
    else:
        line = "".join(
            bars[min(7, max(0, int((v - min_v) / (max_v - min_v) * 7.99)))]
            for v in sample
        )
    return line.rjust(length)


class NonBlockingInputReader:
    """
    Cross-platform non-blocking keyboard input reader supporting Windows, Unix/macOS,
    and programmatic mock queues for automated testing.
    """

    def __init__(self, mock_queue: Optional[List[str]] = None):
        self.mock_queue: List[str] = list(mock_queue) if mock_queue is not None else []
        self._is_windows = sys.platform.startswith("win")
        self._is_tty = sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False

    def push_key(self, key: str) -> None:
        """Injects a simulated keystroke into the queue for deterministic testing."""
        self.mock_queue.append(key)

    def read_key(self) -> Optional[str]:
        """Polls for a single keystroke without blocking. Returns None if no key pressed."""
        if self.mock_queue:
            return self.mock_queue.pop(0)

        if not self._is_tty:
            return None

        if self._is_windows:
            try:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    # Handle special/arrow keys
                    if ch in (b"\x00", b"\xe0"):
                        ch2 = msvcrt.getch()
                        mapping = {
                            b"H": "UP",
                            b"P": "DOWN",
                            b"K": "LEFT",
                            b"M": "RIGHT",
                        }
                        return mapping.get(ch2, None)
                    return ch.decode("utf-8", errors="ignore")
            except Exception:
                return None
        else:
            try:
                import select
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    return sys.stdin.read(1)
            except Exception:
                return None

        return None


@dataclass
class DrawerParameter:
    name: str
    key: str
    value: float
    min_val: float
    max_val: float
    step: float
    format_str: str = "{:.6f}"


class GranularTuningDrawer:
    """State and mutation logic for the interactive `[g]` overlay drawer."""

    def __init__(self, trainer: ResumableTrainer):
        self.trainer = trainer
        self.is_open: bool = False
        self.selected_idx: int = 0
        cfg = trainer.config.training

        self.params: List[DrawerParameter] = [
            DrawerParameter("Learning Rate", "lr", cfg.learning_rate, 1e-6, 0.1, 1.25, "{:.6f}"),
            DrawerParameter("Weight Decay", "wd", cfg.weight_decay, 0.0, 0.1, 1.25, "{:.6f}"),
            DrawerParameter("Prune Rate", "pr", cfg.grokking.prune_rate, 0.01, 0.50, 0.01, "{:.3f}"),
            DrawerParameter("Max Sparsity", "ms", cfg.grokking.max_sparsity, 0.10, 0.95, 0.05, "{:.2f}"),
            DrawerParameter("Slingshot Kick Factor", "sf", cfg.grokking.slingshot_factor, 1.1, 10.0, 0.5, "{:.2f}"),
            DrawerParameter("Grokking Patience", "gp", float(cfg.grokking.grokking_patience), 5.0, 500.0, 5.0, "{:.0f}"),
            DrawerParameter("Grad Clip Norm", "gc", cfg.grad_clip_norm, 0.1, 10.0, 0.5, "{:.2f}"),
        ]

    def select_next(self) -> None:
        self.selected_idx = (self.selected_idx + 1) % len(self.params)

    def select_prev(self) -> None:
        self.selected_idx = (self.selected_idx - 1) % len(self.params)

    def adjust(self, increment: bool) -> str:
        param = self.params[self.selected_idx]
        old_val = param.value
        if param.key in ("lr", "wd"):
            if increment:
                param.value = min(param.max_val, param.value * param.step)
            else:
                param.value = max(param.min_val, param.value / param.step)
        else:
            delta = param.step if increment else -param.step
            param.value = min(param.max_val, max(param.min_val, param.value + delta))

        # Apply immediately to trainer
        if param.key == "lr":
            for pg in self.trainer.optimizer.param_groups:
                pg["lr"] = param.value
        elif param.key == "wd":
            for pg in self.trainer.optimizer.param_groups:
                pg["weight_decay"] = param.value
        elif param.key == "pr":
            self.trainer.config.training.grokking.prune_rate = param.value
        elif param.key == "ms":
            self.trainer.config.training.grokking.max_sparsity = param.value
        elif param.key == "sf":
            self.trainer.config.training.grokking.slingshot_factor = param.value
        elif param.key == "gp":
            self.trainer.config.training.grokking.grokking_patience = int(param.value)
        elif param.key == "gc":
            self.trainer.config.training.grad_clip_norm = param.value

        return f"[*] Drawer tuned {param.name}: {param.format_str.format(old_val)} -> {param.format_str.format(param.value)}"


class InteractiveTrainingStudio:
    """
    Flight Deck Training Studio orchestrating autonomous execution with live tweakability.
    """

    def __init__(
        self,
        trainer: ResumableTrainer,
        kv_cache: Optional[PagedKVCacheManager] = None,
        input_reader: Optional[NonBlockingInputReader] = None,
    ):
        self.trainer = trainer
        self.kv_cache = kv_cache or PagedKVCacheManager(
            num_blocks=64, block_size=16, num_heads=4, head_dim=16
        )
        self.weight_averager = ModelWeightAverager(self.trainer.model, mode="swa")
        self.input_reader = input_reader or NonBlockingInputReader()
        self.drawer = GranularTuningDrawer(trainer)

        self.is_paused: bool = False
        self.is_running: bool = True
        self.current_epoch: int = trainer.start_epoch
        self.current_step: int = trainer.global_step
        self.best_loss: float = trainer.best_loss

        self.history_train_loss: List[float] = []
        self.history_val_loss: List[float] = []
        self.history_weight_norm: List[float] = []
        self.history_sparsity: List[float] = []
        self.events: List[str] = [
            f"[{datetime.now().strftime('%H:%M:%S')}] Flight Deck initialized with autonomous curriculum loop",
            f"[{datetime.now().strftime('%H:%M:%S')}] Model architecture: {self.trainer.model.__class__.__name__}",
        ]
        self.step_delay: float = 0.05
        self.overfit_counter: int = 0
        self.kv_eviction_mode: str = "FIFO"
        self.show_help_overlay: bool = False
        self.last_checkpoint_path: Optional[str] = None

    def log_event(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{ts}] {msg}")
        if len(self.events) > 50:
            self.events.pop(0)

    def save_session_state(self, path: Optional[Path] = None) -> None:
        """Saves active LR, weight decay, current epoch, step, and last checkpoint to session file."""
        import json
        session_file = path or Path("studio_session.json")
        data = {
            "epoch": self.current_epoch,
            "step": self.current_step,
            "learning_rate": self.trainer.optimizer.param_groups[0]["lr"],
            "weight_decay": self.trainer.optimizer.param_groups[0]["weight_decay"],
            "kv_eviction_mode": self.kv_eviction_mode,
            "best_loss": self.best_loss,
            "last_checkpoint_path": str(self.last_checkpoint_path) if self.last_checkpoint_path else None,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)
            self.log_event(f"[*] Session state saved to {session_file.name}")
        except Exception as e:
            self.log_event(f"[-] Failed to save session: {e}")

    def load_session_state(self, path: Optional[Path] = None) -> bool:
        """Restores session parameters from session file."""
        import json
        session_file = path or Path("studio_session.json")
        if not session_file.exists():
            return False
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            lr = data.get("learning_rate")
            if lr is not None:
                for pg in self.trainer.optimizer.param_groups:
                    pg["lr"] = lr
                self.drawer.params[0].value = lr
            wd = data.get("weight_decay")
            if wd is not None:
                for pg in self.trainer.optimizer.param_groups:
                    pg["weight_decay"] = wd
                self.drawer.params[1].value = wd
            mode = data.get("kv_eviction_mode")
            if mode:
                self.kv_eviction_mode = mode
            ckpt = data.get("last_checkpoint_path")
            if ckpt:
                self.last_checkpoint_path = ckpt
            self.log_event(f"[+] Rehydrated session: LR={lr}, WD={wd}, KV_Evict={mode}")
            return True
        except Exception as e:
            self.log_event(f"[-] Error loading session state: {e}")
            return False

    def trigger_slingshot_kick(self) -> None:
        factor = self.trainer.config.training.grokking.slingshot_factor
        self.trainer._apply_slingshot_kick(factor)
        self.log_event(f"HOTKEY [k]: Slingshot Kick applied (factor={factor:.2f}, momentum reset)")

    def trigger_capacity_pruning(self) -> None:
        cfg = self.trainer.config.training.grokking
        new_sparsity = self.trainer._apply_capacity_pruning(cfg.prune_rate, cfg.max_sparsity)
        self.log_event(f"HOTKEY [c]: Capacity Pruned (prune_rate={cfg.prune_rate:.2f}, sparsity={new_sparsity * 100:.1f}%)")

    def trigger_swa_settle(self) -> None:
        self.weight_averager.update()
        self.weight_averager.apply_shadow_weights()
        self.log_event("HOTKEY [w]: SWA weights updated and settled into active model parameters")

    def trigger_nudge_lr(self, up: bool) -> None:
        factor = 1.25 if up else 0.8
        for pg in self.trainer.optimizer.param_groups:
            pg["lr"] *= factor
            cur_lr = pg["lr"]
        self.drawer.params[0].value = cur_lr
        self.save_session_state()
        self.log_event(f"HOTKEY [{'+' if up else '-'}]: Nudged Learning Rate -> {cur_lr:.6f}")

    def trigger_checkpoint_save(self) -> None:
        ckpt_dir = Path(self.trainer.config.checkpoint.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"manual_checkpoint_step_{self.trainer.global_step}.pt"
        save_checkpoint(
            checkpoint_path=path,
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler,
            epoch=self.current_epoch,
            global_step=self.trainer.global_step,
            best_loss=self.trainer.best_loss,
            config_dict=self.trainer.config.__dict__,
        )
        self.last_checkpoint_path = str(path)
        self.save_session_state()
        self.log_event(f"HOTKEY [s]: Forced manual checkpoint exported to {path.name}")

    def toggle_kv_eviction(self) -> None:
        modes = ["FIFO", "LeastRecent", "AttentionSoftmax"]
        cur_idx = modes.index(self.kv_eviction_mode)
        self.kv_eviction_mode = modes[(cur_idx + 1) % len(modes)]
        self.save_session_state()
        self.log_event(f"HOTKEY [e]: PagedAttention KV Cache eviction policy set to {self.kv_eviction_mode}")

    def handle_input(self) -> None:
        key = self.input_reader.read_key()
        if not key:
            return

        if self.show_help_overlay:
            if key in ("?", "h", "q", "\x1b", "\r", "\n"):
                self.show_help_overlay = False
            return

        if self.drawer.is_open:
            if key in ("g", "q", "\x1b"):  # Esc or 'g' or 'q'
                self.drawer.is_open = False
            elif key in ("UP", "k"):
                self.drawer.select_prev()
            elif key in ("DOWN", "j"):
                self.drawer.select_next()
            elif key in ("LEFT", "-"):
                msg = self.drawer.adjust(increment=False)
                self.save_session_state()
                self.log_event(msg)
            elif key in ("RIGHT", "+", "="):
                msg = self.drawer.adjust(increment=True)
                self.save_session_state()
                self.log_event(msg)
            elif key in ("\r", "\n"):
                self.drawer.is_open = False
                self.save_session_state()
        else:
            if key in ("q", "\x03"):  # 'q' or Ctrl-C
                self.is_running = False
                self.save_session_state()
                self.log_event("Session terminated by user [q]")
            elif key in ("?", "h"):
                self.show_help_overlay = True
            elif key == "g":
                self.drawer.is_open = True
            elif key in (" ", "p"):
                self.is_paused = not self.is_paused
                self.log_event(f"Autonomous curriculum loop {'PAUSED' if self.is_paused else 'RESUMED'} [Space]")
            elif key == "k":
                self.trigger_slingshot_kick()
            elif key == "c":
                self.trigger_capacity_pruning()
            elif key == "w":
                self.trigger_swa_settle()
            elif key in ("+", "="):
                self.trigger_nudge_lr(up=True)
            elif key == "-":
                self.trigger_nudge_lr(up=False)
            elif key == "e":
                self.toggle_kv_eviction()
            elif key == "s":
                self.trigger_checkpoint_save()

    def step(self) -> Optional[Dict[str, float]]:
        """Executes one autonomous curriculum epoch/step, handling input and telemetry."""
        self.handle_input()
        if self.is_paused or not self.is_running:
            return None

        # Execute single training epoch
        train_loss = self.trainer.train_epoch(self.current_epoch)
        val_loss = self.trainer.evaluate() if self.trainer.val_loader else train_loss

        self.current_epoch += 1
        self.current_step = self.trainer.global_step

        weight_norm = self.trainer._compute_weight_norm()
        sparsity = self.trainer._compute_parameter_sparsity()

        self.history_train_loss.append(train_loss)
        self.history_val_loss.append(val_loss)
        self.history_weight_norm.append(weight_norm)
        self.history_sparsity.append(sparsity)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.trainer.best_loss = val_loss

        # Autonomous Grokking and Pruning Loop
        grok_cfg = self.trainer.config.training.grokking
        if grok_cfg.enabled:
            if train_loss < grok_cfg.prune_threshold:
                self.overfit_counter += 1
                if grok_cfg.capacity_pruning and (self.current_epoch % grok_cfg.prune_interval_epochs == 0):
                    new_sp = self.trainer._apply_capacity_pruning(grok_cfg.prune_rate, grok_cfg.max_sparsity)
                    self.log_event(f"[*] Autonomous Pruning pass: Sparsity reached {new_sp * 100:.1f}%")
                if grok_cfg.slingshot_enabled and (self.overfit_counter % grok_cfg.slingshot_interval == 0):
                    self.trainer._apply_slingshot_kick(grok_cfg.slingshot_factor)
                    self.log_event(f"[*] Autonomous Slingshot Kick triggered at epoch {self.current_epoch}")
            else:
                self.overfit_counter = 0

        # Simulate dynamic PagedAttention token generation
        seq_id = self.current_epoch % 4
        self.kv_cache.allocate_sequence(seq_id, initial_tokens=1)
        fake_k = torch.randn(self.kv_cache.num_heads, self.kv_cache.head_dim)
        fake_v = torch.randn(self.kv_cache.num_heads, self.kv_cache.head_dim)
        self.kv_cache.append_token(seq_id, fake_k, fake_v)

        entry = {
            "epoch": self.current_epoch - 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "weight_norm": weight_norm,
            "sparsity": sparsity,
        }
        self.trainer.history.append(entry)
        return entry

    def get_curriculum_phase(self) -> str:
        """Returns the current curriculum/grokking phase description."""
        if not self.history_train_loss:
            return "Init"
        train_loss = self.history_train_loss[-1]
        val_loss = self.history_val_loss[-1] if self.history_val_loss else train_loss
        grok_cfg = self.trainer.config.training.grokking
        prune_thresh = grok_cfg.prune_threshold if (grok_cfg and grok_cfg.enabled) else 0.05
        if self.overfit_counter > 0 or (train_loss <= prune_thresh and val_loss > prune_thresh * 2):
            return "Grokking Transition"
        elif train_loss <= prune_thresh and val_loss <= prune_thresh * 2:
            return "Generalization"
        else:
            return "Memorization"

    def render(self, clear_screen: bool = True) -> str:
        """Renders the comprehensive 4-quadrant layout + tuning drawer + footer, or help overlay."""
        import re

        width = 86
        lines = []

        if clear_screen:
            lines.append("\033[H\033[J")

        if self.show_help_overlay:
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}┌{'─' * (width - 2)}┐{COLOR_RESET}")
            title = "SPODEIAN-ML FLIGHT DECK STUDIO — HELP & TELEMETRY OVERLAY"
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}  {COLOR_BOLD}{COLOR_YELLOW}{title}{COLOR_RESET}".ljust(width + 12) + f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}├{'─' * (width - 2)}┤{COLOR_RESET}")
            help_rows = [
                f" {COLOR_BOLD}COMMAND SHORTCUTS:{COLOR_RESET}",
                f"   {COLOR_WHITE}[Space] / [p]{COLOR_RESET}  : Toggle Pause / Autonomous Execution",
                f"   {COLOR_WHITE}[g]{COLOR_RESET}            : Open / Close Granular Tuning Drawer",
                f"   {COLOR_WHITE}[↑/k] / [↓/j]{COLOR_RESET}  : Navigate Drawer Parameters (LR, WD, Prune, Slingshot)",
                f"   {COLOR_WHITE}[←/-] / [→/+]{COLOR_RESET}  : Decrement / Increment Selected Parameter Value",
                f"   {COLOR_WHITE}[Enter]{COLOR_RESET}        : Confirm Drawer Tuning & Apply to Optimizer",
                f"   {COLOR_WHITE}[k]{COLOR_RESET}            : Apply Slingshot Kick (Gradient Perturbation)",
                f"   {COLOR_WHITE}[c]{COLOR_RESET}            : Apply Capacity Pruning (Weight Magnitude Sparsity)",
                f"   {COLOR_WHITE}[w]{COLOR_RESET}            : Settle SWA Shadow Weights into Active Model",
                f"   {COLOR_WHITE}[+] / [-]{COLOR_RESET}      : Quick Nudge Active Learning Rate (±25%)",
                f"   {COLOR_WHITE}[e]{COLOR_RESET}            : Cycle PagedAttention Eviction (FIFO/LeastRecent/Softmax)",
                f"   {COLOR_WHITE}[s]{COLOR_RESET}            : Manual Force Save Resumable Checkpoint",
                f"   {COLOR_WHITE}[?] / [h]{COLOR_RESET}      : Toggle this Help & Telemetry Overlay",
                f"   {COLOR_WHITE}[q]{COLOR_RESET}            : Clean Exit and Persist Session State",
                f"",
                f" {COLOR_BOLD}ACTIVE ENGINE PARAMETERS & TELEMETRY:{COLOR_RESET}",
                f"   Active LR:        {self.trainer.optimizer.param_groups[0]['lr']:.6f}   Weight Decay: {self.trainer.optimizer.param_groups[0]['weight_decay']:.5f}",
                f"   KV Eviction Mode: {self.kv_eviction_mode:<10}  KV Usage:     {self.kv_cache.num_blocks - len(self.kv_cache.free_blocks)}/{self.kv_cache.num_blocks} blocks",
                f"   Curriculum Phase: {self.get_curriculum_phase()}",
                f"   Last Checkpoint:  {Path(self.last_checkpoint_path).name if self.last_checkpoint_path else 'None (auto-saves on hotkey [s])'}",
            ]
            for row in help_rows:
                plain_len = len(re.sub(r"\033\[[0-9;]*m", "", row))
                pad = max(0, width - 2 - plain_len)
                lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row}{' ' * pad}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}├{'─' * (width - 2)}┤{COLOR_RESET}")
            dismiss_msg = " Press [?] or [h] or [q] or [Esc] or [Enter] to dismiss this overlay"
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{COLOR_DIM}{dismiss_msg:<{width - 2}}{COLOR_RESET}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}└{'─' * (width - 2)}┘{COLOR_RESET}")
            return "\n".join(lines)

        # Header with active curriculum badge
        total_epochs = self.trainer.config.training.max_epochs
        phase_str = self.get_curriculum_phase()
        status_tag = f"{COLOR_YELLOW}[PAUSED]{COLOR_RESET}" if self.is_paused else f"{COLOR_GREEN}[RUNNING]{COLOR_RESET}"
        plain_status = "[PAUSED]" if self.is_paused else "[RUNNING]"
        plain_title = f"SPODEIAN-ML · Studio ► Ep {self.current_epoch}/{total_epochs} | {phase_str}"
        plain_len = len(f"  {plain_title} {plain_status}")
        pad = max(0, width - 2 - plain_len)

        header = f"{COLOR_BOLD}{COLOR_CYAN}┌{'─' * (width - 2)}┐{COLOR_RESET}\n"
        header += f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}  {COLOR_BOLD}{plain_title}{COLOR_RESET} {status_tag}{' ' * pad}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}\n"
        header += f"{COLOR_BOLD}{COLOR_CYAN}├{'─' * 41}┬{'─' * (width - 44)}┤{COLOR_RESET}"
        lines.append(header)

        # Telemetry variables
        cur_train = self.history_train_loss[-1] if self.history_train_loss else 0.0
        cur_val = self.history_val_loss[-1] if self.history_val_loss else 0.0
        gen_gap = abs(cur_train - cur_val)
        train_spark = format_sparkline(self.history_train_loss, 16)
        val_spark = format_sparkline(self.history_val_loss, 16)

        cur_norm = self.history_weight_norm[-1] if self.history_weight_norm else 0.0
        cur_sparsity = self.history_sparsity[-1] if self.history_sparsity else 0.0
        norm_spark = format_sparkline(self.history_weight_norm, 16)

        # Quadrant 1 & 2 Rows
        row1_q1 = f" Train Loss: {cur_train:.5f}  [{train_spark}]"
        row1_q2 = f" L2 Weight Norm: {cur_norm:.3f} [{norm_spark}]"
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row1_q1:<41}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row1_q2:<42}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")

        row2_q1 = f" Val Loss:   {cur_val:.5f}  [{val_spark}]"
        row2_q2 = f" Weight Sparsity: {cur_sparsity * 100:>5.1f}%"
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row2_q1:<41}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row2_q2:<42}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")

        row3_q1 = f" Gen Gap:    {gen_gap:.5f} (Best: {self.best_loss:.4f})"
        row3_q2 = f" Grokking Overfit Patience: {self.overfit_counter}/" + str(self.trainer.config.training.grokking.grokking_patience)
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row3_q1:<41}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row3_q2:<42}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")

        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}├{'─' * 41}┼{'─' * (width - 44)}┤{COLOR_RESET}")

        # Quadrant 3 & 4 Rows
        lr = self.trainer.optimizer.param_groups[0]["lr"]
        wd = self.trainer.optimizer.param_groups[0]["weight_decay"]
        allocated_blocks = self.kv_cache.num_blocks - len(self.kv_cache.free_blocks)
        kv_util = allocated_blocks / max(1, self.kv_cache.num_blocks) * 100

        recent_events = self.events[-3:] if len(self.events) >= 3 else self.events + [""] * (3 - len(self.events))

        row4_q3 = f" LR: {lr:.6f} | WD: {wd:.5f}"
        row4_q4 = f" Event: {recent_events[0][:32]}"
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row4_q3:<41}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row4_q4:<42}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")

        row5_q3 = f" KV Cache: {allocated_blocks}/{self.kv_cache.num_blocks} blks ({kv_util:.0f}%) [{self.kv_eviction_mode}]"
        row5_q4 = f" Event: {recent_events[1][:32]}"
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row5_q3:<41}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row5_q4:<42}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")

        row6_q3 = f" Epoch: {self.current_epoch} | Step: {self.current_step}"
        row6_q4 = f" Event: {recent_events[2][:32]}"
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row6_q3:<41}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{row6_q4:<42}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")

        # Drawer Overlay or Separator
        if self.drawer.is_open:
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}├{'─' * (width - 2)}┤{COLOR_RESET}")
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}  {COLOR_BOLD}{COLOR_YELLOW}GRANULAR TUNING DRAWER [g]{COLOR_RESET} (↑/↓ Select, ←/→ Nudge, Enter Confirm)".ljust(width + 12) + f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")
            for idx, p in enumerate(self.drawer.params):
                prefix = "  ▶ " if idx == self.drawer.selected_idx else "    "
                val_repr = p.format_str.format(p.value)
                line_str = f"{prefix}{p.name:<24}: {val_repr:>10}  (Min: {p.format_str.format(p.min_val)}, Max: {p.format_str.format(p.max_val)})"
                if idx == self.drawer.selected_idx:
                    formatted_line = f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{COLOR_BOLD}{COLOR_WHITE}{line_str:<{width - 2}}{COLOR_RESET}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}"
                else:
                    formatted_line = f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}{line_str:<{width - 2}}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}"
                lines.append(formatted_line)

        # Footer
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}├{'─' * (width - 2)}┤{COLOR_RESET}")
        hotkeys = (
            f" [Space] Pause | [g] Tuning Drawer | [k] Slingshot | [c] Prune | [w] SWA\n"
            f" [+] / [-] Nudge LR | [e] KV Evict | [s] Checkpoint | [?] Help | [q] Quit"
        )
        for h_line in hotkeys.split("\n"):
            lines.append(f"{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET} {COLOR_DIM}{h_line:<{width - 3}}{COLOR_RESET}{COLOR_BOLD}{COLOR_CYAN}│{COLOR_RESET}")
        lines.append(f"{COLOR_BOLD}{COLOR_CYAN}└{'─' * (width - 2)}┘{COLOR_RESET}")

        return "\n".join(lines)


def run_interactive_training(
    trainer: ResumableTrainer,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    headless: bool = False,
    resume: bool = False,
    input_reader: Optional[NonBlockingInputReader] = None,
    render_interval: float = 0.05,
) -> List[Dict[str, float]]:
    """
    Main interactive entrypoint for autonomous training with live steering.
    """
    studio = InteractiveTrainingStudio(trainer, input_reader=input_reader)
    if resume:
        studio.load_session_state()

    target_epochs = max_epochs or trainer.config.training.max_epochs

    try:
        while studio.is_running and studio.current_epoch < target_epochs:
            entry = studio.step()

            if not headless:
                frame = studio.render(clear_screen=True)
                sys.stdout.write(frame + "\n")
                sys.stdout.flush()

            if max_steps is not None and studio.current_step >= max_steps:
                break

            time.sleep(render_interval)

    except KeyboardInterrupt:
        studio.log_event("Interrupted by user via SIGINT")
    finally:
        studio.save_session_state()
        if not headless:
            sys.stdout.write(f"\n{COLOR_GREEN}[+] Interactive training session concluded.{COLOR_RESET}\n")
            sys.stdout.flush()

    return trainer.history
