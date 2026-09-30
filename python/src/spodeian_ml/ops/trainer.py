"""
Universal Deterministic Resumable Trainer for spodeian-ml.
Enforces bit-perfect resumption, gradient clipping, periodic checkpointing,
and value-aware loss feedback reporting to the Spodeian Data Lake.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from spodeian_ml.ops.config import ExperimentConfig
from spodeian_ml.ops.checkpoint import save_checkpoint, load_checkpoint


class ResumableTrainer:
    """
    State-managed training loop guaranteeing exact reproducibility and recovery across sessions.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.training.max_epochs),
            eta_min=1e-6,
        )

        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.history: List[Dict[str, float]] = []

        # Automatic Resume if requested in the Git-versioned config
        if self.config.checkpoint.resume_from:
            self._resume(Path(self.config.checkpoint.resume_from))

    def _resume(self, checkpoint_path: Path) -> None:
        """Restores exact state from previous run."""
        epoch, step, best_loss, _ = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.start_epoch = epoch + 1
        self.global_step = step
        self.best_loss = best_loss

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            self.optimizer.zero_grad()

            # Handle SSL models vs Supervised models
            latent = None
            if hasattr(self.model, "compute_ssl_loss"):
                seed = self.config.seed + self.global_step
                loss, latent = self.model.compute_ssl_loss(x, seed=seed)
            else:
                out = self.model(x)
                grokking_cfg = self.config.training.grokking
                if grokking_cfg.enabled and grokking_cfg.dynamic_sample_dropping and x.dim() >= 2:
                    sample_losses = nn.functional.mse_loss(out, x, reduction="none").mean(
                        dim=list(range(1, out.dim()))
                    )
                    hard_mask = sample_losses > grokking_cfg.drop_loss_threshold
                    if hard_mask.sum() > 0:
                        loss = sample_losses[hard_mask].mean()
                    else:
                        loss = sample_losses.mean()
                else:
                    loss = nn.functional.mse_loss(out, x)
                latent = out

            # Activation Sparsity Regularization (L1 penalty on activations/representations)
            if self.config.training.grokking.enabled and self.config.training.grokking.activation_sparsity_penalty > 0.0:
                if latent is not None:
                    loss = loss + self.config.training.grokking.activation_sparsity_penalty * latent.abs().mean()

            loss.backward()

            # Gradient clipping to protect against exploding gradients
            if self.config.training.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        self.scheduler.step()
        return total_loss / max(1, num_batches)

    def evaluate(self) -> float:
        if not self.val_loader:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)

                if hasattr(self.model, "compute_ssl_loss"):
                    loss, _ = self.model.compute_ssl_loss(x, seed=self.config.seed)
                else:
                    out = self.model(x)
                    loss = nn.functional.mse_loss(out, x)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def _compute_weight_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                total_norm += p.norm(2).item() ** 2
        return total_norm ** 0.5

    def _compute_parameter_sparsity(self) -> float:
        """Computes fraction of structural zero parameters across active weights."""
        total_params = 0
        zero_params = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad and ("weight" in name):
                total_params += p.numel()
                zero_params += (p == 0).sum().item()
        return zero_params / max(1, total_params)

    def _apply_capacity_pruning(self, prune_rate: float, max_sparsity: float) -> float:
        """Prunes smallest magnitude weights (L1 pruning) across linear/conv weights up to max_sparsity."""
        current_sparsity = self._compute_parameter_sparsity()
        if current_sparsity >= max_sparsity:
            return current_sparsity

        all_active_weights = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and ("weight" in name) and p.dim() >= 2:
                active = p.data[p.data != 0].abs()
                if active.numel() > 0:
                    all_active_weights.append(active)

        if not all_active_weights:
            return current_sparsity

        concat_weights = torch.cat(all_active_weights)
        k = max(1, int(concat_weights.numel() * prune_rate))
        if k >= concat_weights.numel():
            return current_sparsity

        threshold, _ = torch.kthvalue(concat_weights, k)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad and ("weight" in name) and p.dim() >= 2:
                    mask = p.abs() > threshold
                    p.data.mul_(mask.to(p.dtype))

        return self._compute_parameter_sparsity()

    def _apply_slingshot_kick(self, factor: float) -> None:
        """Executes cyclic LR kick and resets AdamW momentum to dislodge weights from memorization basin."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        # Reset optimizer momentum states
        for p in self.model.parameters():
            state = self.optimizer.state.get(p)
            if state:
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()

    def run(self) -> List[Dict[str, float]]:
        import time
        import uuid
        from spodeian_ml.ops.experiment import (
            ExperimentRecord,
            save_experiment_record,
            compute_file_sha256,
            get_system_hardware_context,
        )

        start_time = time.time()
        run_id = str(uuid.uuid4())
        checkpoint_dir = Path(self.config.checkpoint.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        overfit_patience_counter = 0
        grokking_cfg = self.config.training.grokking
        grokking_mode = grokking_cfg.enabled
        max_epochs = self.config.training.max_epochs

        # In grokking mode, apply grokking weight decay override
        if grokking_mode:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = grokking_cfg.l2_regularization

        for epoch in range(self.start_epoch, max_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate() if self.val_loader else train_loss
            
            history_entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            
            if grokking_mode and grokking_cfg.track_weight_norms:
                history_entry["weight_norm"] = self._compute_weight_norm()
            if grokking_mode and grokking_cfg.track_representation_sparsity:
                history_entry["sparsity"] = self._compute_parameter_sparsity()

            self.history.append(history_entry)

            # Grokking tracking & hybrid sparsity mechanics
            if grokking_mode:
                if train_loss < grokking_cfg.prune_threshold:
                    overfit_patience_counter += 1

                    # Dynamic Capacity Pruning (The Lottery Ticket Approach)
                    if grokking_cfg.capacity_pruning and (epoch % grokking_cfg.prune_interval_epochs == 0):
                        new_sparsity = self._apply_capacity_pruning(
                            prune_rate=grokking_cfg.prune_rate,
                            max_sparsity=grokking_cfg.max_sparsity,
                        )
                        history_entry["pruned_sparsity"] = new_sparsity

                    # The Slingshot Mechanism
                    if grokking_cfg.slingshot_enabled and (overfit_patience_counter % grokking_cfg.slingshot_interval == 0):
                        self._apply_slingshot_kick(grokking_cfg.slingshot_factor)
                        history_entry["slingshot_applied"] = 1.0

                    if overfit_patience_counter > grokking_cfg.grokking_patience:
                        print(f"Stopping: Grokking patience exceeded at epoch {epoch}")
                        break
                else:
                    overfit_patience_counter = 0

            # Save best checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_checkpoint(
                    checkpoint_path=checkpoint_dir / "best_model.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_loss=self.best_loss,
                    config_dict=self.config.__dict__,
                )

            # Save periodic checkpoint
            if (epoch + 1) % self.config.checkpoint.save_every_n_epochs == 0:
                save_checkpoint(
                    checkpoint_path=checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_loss=self.best_loss,
                    config_dict=self.config.__dict__,
                )

        # Emit experiment record to the Spodeian Data Lake Registry
        try:
            duration = time.time() - start_time
            best_ckpt = checkpoint_dir / "best_model.pt"
            ckpt_path_str = str(best_ckpt) if best_ckpt.exists() else None
            ckpt_sha = compute_file_sha256(best_ckpt) if best_ckpt.exists() else None

            final_metrics = {
                "epochs_run": float(len(self.history)),
                "best_loss": float(self.best_loss),
            }
            if self.history:
                final_metrics["final_train_loss"] = float(self.history[-1]["train_loss"])
                final_metrics["final_val_loss"] = float(self.history[-1]["val_loss"])
                if "weight_norm" in self.history[-1]:
                    final_metrics["final_weight_norm"] = float(self.history[-1]["weight_norm"])
                if "sparsity" in self.history[-1]:
                    final_metrics["final_sparsity"] = float(self.history[-1]["sparsity"])

            import yaml
            cfg_yaml = yaml.dump(self.config.__dict__, default_flow_style=False)

            record = ExperimentRecord(
                run_id=run_id,
                experiment_name=self.config.experiment_name,
                config_yaml_snapshot=cfg_yaml,
                final_metrics=final_metrics,
                checkpoint_path=ckpt_path_str,
                checkpoint_sha256=ckpt_sha,
                hardware_context=get_system_hardware_context(),
                timestamp_epoch_s=int(time.time()),
                duration_seconds=duration,
                completion_status="COMPLETED",
                tags=["training", "resumable_trainer"],
            )
            save_experiment_record(record)
        except Exception:
            pass

        return self.history
