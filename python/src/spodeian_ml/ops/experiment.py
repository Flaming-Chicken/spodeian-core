"""
Unified Experiment Registry & Benchmark Tracking for spodeian-ml and spodeian-lake.
Provides bit-perfect experiment tracking, config snapshots, hardware context capture,
and side-by-side comparison tables.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
import time
import json
import hashlib
import platform
import torch

from spodeian_ml.data.lake_loader import get_default_lake_path


@dataclass
class ExperimentRecord:
    """Metadata record for a training run or benchmark in the Spodeian Data Lake."""
    run_id: str
    experiment_name: str
    config_yaml_snapshot: str
    final_metrics: Dict[str, float]
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    hardware_context: Dict[str, str] = field(default_factory=dict)
    timestamp_epoch_s: int = field(default_factory=lambda: int(time.time()))
    duration_seconds: float = 0.0
    completion_status: str = "COMPLETED"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        return cls(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
            config_yaml_snapshot=data.get("config_yaml_snapshot", ""),
            final_metrics=data.get("final_metrics", {}),
            checkpoint_path=data.get("checkpoint_path"),
            checkpoint_sha256=data.get("checkpoint_sha256"),
            hardware_context=data.get("hardware_context", {}),
            timestamp_epoch_s=data.get("timestamp_epoch_s", int(time.time())),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            completion_status=data.get("completion_status", "COMPLETED"),
            tags=data.get("tags", []),
        )


def compute_file_sha256(path: Path) -> Optional[str]:
    """Calculates SHA-256 checksum of a file on disk."""
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_system_hardware_context() -> Dict[str, str]:
    """Captures runtime hardware and framework versions."""
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple Silicon MPS"

    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": device_name,
        "cuda_available": str(torch.cuda.is_available()),
    }


def save_experiment_record(record: ExperimentRecord, lake_dir: Optional[Path] = None) -> Path:
    """
    Saves an experiment record into lake_registry.json under the 'experiments' key.
    Preserves existing partitions and experiments without data loss.
    """
    path = lake_dir or get_default_lake_path()
    path.mkdir(parents=True, exist_ok=True)
    registry_file = path / "lake_registry.json"

    data: Dict[str, Any] = {"partitions": {}, "experiments": {}}
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"partitions": {}, "experiments": {}}

    if "experiments" not in data:
        data["experiments"] = {}

    data["experiments"][record.run_id] = record.to_dict()

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return registry_file


def load_experiment_registry(lake_dir: Optional[Path] = None) -> List[ExperimentRecord]:
    """Loads all experiment records from lake_registry.json sorted by timestamp descending."""
    path = lake_dir or get_default_lake_path()
    registry_file = path / "lake_registry.json"
    if not registry_file.exists():
        return []

    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    experiments_raw = data.get("experiments", {})
    records = []
    for exp_dict in experiments_raw.values():
        records.append(ExperimentRecord.from_dict(exp_dict))

    records.sort(key=lambda r: r.timestamp_epoch_s, reverse=True)
    return records


def get_experiment_by_id(run_id: str, lake_dir: Optional[Path] = None) -> Optional[ExperimentRecord]:
    """Finds an experiment record matching run_id or short run_id prefix."""
    records = load_experiment_registry(lake_dir)
    for r in records:
        if r.run_id == run_id or r.run_id.startswith(run_id):
            return r
    return None


def format_experiment_list_table(experiments: List[ExperimentRecord]) -> str:
    """Renders a formatted ASCII table of experiment records."""
    if not experiments:
        return "No experiments found in lake registry."

    header = f"{'Run ID':<24} | {'Experiment Name':<24} | {'Status':<11} | {'Epochs':<6} | {'Val Loss':<10} | {'Best Loss':<10} | {'Timestamp':<19}"
    sep = "-" * len(header)
    rows = [header, sep]

    for exp in experiments:
        short_id = exp.run_id[:22] + ".." if len(exp.run_id) > 24 else exp.run_id
        epochs = str(int(exp.final_metrics.get("epochs_run", exp.final_metrics.get("total_epochs", 0))))
        val_loss = exp.final_metrics.get("val_loss")
        val_str = f"{val_loss:.5f}" if val_loss is not None else "N/A"
        best_loss = exp.final_metrics.get("best_loss")
        best_str = f"{best_loss:.5f}" if best_loss is not None else "N/A"
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp.timestamp_epoch_s))

        rows.append(
            f"{short_id:<24} | {exp.experiment_name[:24]:<24} | {exp.completion_status:<11} | {epochs:<6} | {val_str:<10} | {best_str:<10} | {ts_str:<19}"
        )

    return "\n".join(rows)


def format_experiment_comparison_table(exp_a: ExperimentRecord, exp_b: ExperimentRecord) -> str:
    """Renders a side-by-side diff table comparing metrics and parameters between two runs."""
    id_a = exp_a.run_id[:8]
    id_b = exp_b.run_id[:8]

    header = f"{'Metric / Property':<28} | {f'Run A ({id_a})':<20} | {f'Run B ({id_b})':<20} | {'Delta (B - A)':<16}"
    sep = "-" * len(header)
    rows = [
        f"=== EXPERIMENT COMPARISON ===",
        f"Run A: {exp_a.experiment_name} [{exp_a.run_id}] ({exp_a.completion_status})",
        f"Run B: {exp_b.experiment_name} [{exp_b.run_id}] ({exp_b.completion_status})",
        "",
        header,
        sep,
    ]

    # Compare metadata
    rows.append(f"{'Duration (seconds)':<28} | {exp_a.duration_seconds:<20.2f} | {exp_b.duration_seconds:<20.2f} | {exp_b.duration_seconds - exp_a.duration_seconds:<+16.2f}")
    
    # Compare all metrics present in either run
    all_metric_keys = sorted(list(set(exp_a.final_metrics.keys()) | set(exp_b.final_metrics.keys())))
    for key in all_metric_keys:
        val_a = exp_a.final_metrics.get(key)
        val_b = exp_b.final_metrics.get(key)
        val_a_str = f"{val_a:.6f}" if val_a is not None else "N/A"
        val_b_str = f"{val_b:.6f}" if val_b is not None else "N/A"
        if val_a is not None and val_b is not None:
            delta_str = f"{val_b - val_a:+.6f}"
        else:
            delta_str = "N/A"
        rows.append(f"{key:<28} | {val_a_str:<20} | {val_b_str:<20} | {delta_str:<16}")

    # Compare hardware / environment
    rows.append(sep)
    hw_keys = sorted(list(set(exp_a.hardware_context.keys()) | set(exp_b.hardware_context.keys())))
    for key in hw_keys:
        hw_a = exp_a.hardware_context.get(key, "N/A")
        hw_b = exp_b.hardware_context.get(key, "N/A")
        match = "MATCH" if hw_a == hw_b else "DIFF"
        rows.append(f"{f'env:{key}':<28} | {hw_a:<20} | {hw_b:<20} | {match:<16}")

    return "\n".join(rows)
