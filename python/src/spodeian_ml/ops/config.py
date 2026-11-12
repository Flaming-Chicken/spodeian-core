"""
Declarative Git-Versioned Configuration Parser for spodeian-ml.
Enforces strict schema validation for models, SSL pretext tasks, and resumable training loops.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class ModelConfig:
    encoder_type: str = "transformer"  # "transformer", "mlp", "gru"
    input_dim: int = 16
    hidden_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    head_type: str = "mlp"  # "linear_probe", "mlp", "temporal", "none"


@dataclass
class SSLConfig:
    task: str = "masked_autoencoder"  # "masked_autoencoder", "contrastive_infonce"
    mask_ratio: float = 0.25
    mask_value: float = 0.0
    temperature: float = 0.07  # InfoNCE temperature


@dataclass
class GrokkingConfig:
    enabled: bool = False
    grokking_patience: int = 10000  # Epochs to wait AFTER training loss hits ~0 before giving up
    l2_regularization: float = 1e-2 # High weight decay is often essential for grokking
    track_weight_norms: bool = True
    track_representation_sparsity: bool = True
    # Capacity Pruning (The Lottery Ticket Approach)
    capacity_pruning: bool = False
    prune_threshold: float = 1e-3
    prune_rate: float = 0.05
    max_sparsity: float = 0.70
    prune_interval_epochs: int = 5
    # Dynamic Sample Dropping (Curriculum Eviction)
    dynamic_sample_dropping: bool = False
    drop_loss_threshold: float = 1e-4
    # The Slingshot Mechanism
    slingshot_enabled: bool = False
    slingshot_interval: int = 25
    slingshot_factor: float = 4.0
    # Activation Sparsity Penalty (L1 on latent activations)
    activation_sparsity_penalty: float = 0.0

@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 5
    grokking: GrokkingConfig = field(default_factory=GrokkingConfig)


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    save_every_n_steps: Optional[int] = None
    resume_from: Optional[str] = None  # Path to bit-perfect checkpoint state


@dataclass
class DataLakeConfig:
    partition_paths: List[str] = field(default_factory=list)
    dataset_id: str = "default_dataset"
    feedback_reporting: bool = True  # Reports batch loss back to spodeian-lake for value-aware retention


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataLakeConfig = field(default_factory=DataLakeConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        model_cfg = ModelConfig(**raw.get("model", {}))
        ssl_cfg = SSLConfig(**raw.get("ssl", {}))
        train_raw = raw.get("training", {})
        grokking_raw = train_raw.pop("grokking", {})
        if isinstance(grokking_raw, dict):
            grokking_cfg = GrokkingConfig(**grokking_raw)
        elif isinstance(grokking_raw, GrokkingConfig):
            grokking_cfg = grokking_raw
        else:
            grokking_cfg = GrokkingConfig()
        train_cfg = TrainingConfig(**train_raw, grokking=grokking_cfg)
        chkpt_cfg = CheckpointConfig(**raw.get("checkpoint", {}))
        data_cfg = DataLakeConfig(**raw.get("data", {}))

        return cls(
            experiment_name=raw.get("experiment_name", "unnamed_experiment"),
            seed=raw.get("seed", 42),
            model=model_cfg,
            ssl=ssl_cfg,
            training=train_cfg,
            checkpoint=chkpt_cfg,
            data=data_cfg,
        )

    def to_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
