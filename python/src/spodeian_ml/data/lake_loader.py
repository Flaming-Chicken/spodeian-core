"""
Windows Data Lake Partition Loader.
Streams Parquet partitions written by spodeian-lake directly into PyTorch Dataset & DataLoader objects.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import json
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class SpodeianLakeDataset(Dataset):
    """PyTorch Dataset that streams columnar tensors from a local Parquet partition."""

    def __init__(self, parquet_path: Path, feature_columns: Optional[List[str]] = None):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet partition not found: {self.parquet_path}")

        table = pq.read_table(self.parquet_path)
        self.column_names = table.column_names

        # Select numeric feature columns
        if feature_columns:
            self.feature_columns = [col for col in feature_columns if col in self.column_names]
        else:
            self.feature_columns = [
                name for name, field in zip(table.schema.names, table.schema)
                if str(field.type).startswith(("float", "double", "int"))
            ]

        # Convert table to PyTorch tensor matrix (rows x features)
        tensors = []
        for col_name in self.feature_columns:
            arr = table[col_name].to_numpy().copy()
            tensors.append(torch.from_numpy(arr).float().unsqueeze(1))

        if tensors:
            self.features = torch.cat(tensors, dim=1)
        else:
            self.features = torch.empty((table.num_rows, 0))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]


def get_default_lake_path() -> Path:
    """Resolves default Windows Data Lake directory (%LOCALAPPDATA%/spodeian/lake)."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "spodeian" / "lake"
    return Path(".spodeian") / "lake"


def load_lake_registry(lake_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Reads the metadata registry of the local Windows Data Lake."""
    path = lake_dir or get_default_lake_path()
    registry_file = path / "lake_registry.json"
    if not registry_file.exists():
        return {"partitions": {}}
    with open(registry_file, "r", encoding="utf-8") as f:
        return json.load(f)
