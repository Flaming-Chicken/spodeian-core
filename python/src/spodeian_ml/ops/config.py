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
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 5


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
        train_cfg = TrainingConfig(**raw.get("training", {}))
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
