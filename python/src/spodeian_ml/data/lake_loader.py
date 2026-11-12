"""
Windows Data Lake Partition Loader.
Streams Parquet partitions written by spodeian-lake directly into PyTorch Dataset & DataLoader objects.
Includes automatic registry discovery, chronological partition ordering, and SHA-256 integrity verification.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import os
import json
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


class SpodeianLakeDataset(Dataset):
    """PyTorch Dataset that streams columnar tensors from local Parquet partitions."""

    def __init__(
        self,
        parquet_path: Union[str, Path, List[Union[str, Path]]],
        feature_columns: Optional[List[str]] = None,
        partition_metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        self.partition_metadata: List[Dict[str, Any]] = partition_metadata or []
        self.files: List[Path] = []

        if isinstance(parquet_path, (list, tuple)):
            for p in parquet_path:
                path_obj = Path(p)
                if path_obj.is_dir():
                    self.files.extend(sorted(path_obj.glob("*.parquet")))
                elif path_obj.exists():
                    self.files.append(path_obj)
                else:
                    raise FileNotFoundError(f"Parquet partition not found: {path_obj}")
        else:
            path_obj = Path(parquet_path)
            if path_obj.is_dir():
                self.files = sorted(path_obj.glob("*.parquet"))
                if not self.files:
                    raise FileNotFoundError(f"No .parquet files found in directory: {path_obj}")
            elif path_obj.exists():
                self.files = [path_obj]
            else:
                raise FileNotFoundError(f"Parquet partition not found: {path_obj}")

        if not self.files:
            raise FileNotFoundError(f"No valid Parquet files found for: {parquet_path}")

        # Read tables and concatenate
        tables = [pq.read_table(f) for f in self.files]
        if len(tables) == 1:
            combined_table = tables[0]
        else:
            combined_table = pa.concat_tables(tables)

        self.column_names = combined_table.column_names

        # Select numeric feature columns
        if feature_columns:
            self.feature_columns = [col for col in feature_columns if col in self.column_names]
        else:
            self.feature_columns = [
                name for name, field in zip(combined_table.schema.names, combined_table.schema)
                if str(field.type).startswith(("float", "double", "int"))
            ]

        # Convert table to PyTorch tensor matrix (rows x features)
        tensors = []
        for col_name in self.feature_columns:
            arr = combined_table[col_name].to_numpy().copy()
            tensors.append(torch.from_numpy(arr).float().unsqueeze(1))

        if tensors:
            self.features = torch.cat(tensors, dim=1)
        else:
            self.features = torch.empty((combined_table.num_rows, 0))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]

    @classmethod
    def from_registry(
        cls,
        dataset_id: str,
        lake_dir: Optional[Union[str, Path]] = None,
        feature_columns: Optional[List[str]] = None,
        verify_sha256: bool = True,
    ) -> "SpodeianLakeDataset":
        """
        Resolves a named dataset from the LakeRegistry, sorts its partitions chronologically,
        validates SHA-256 integrity checksums, and loads them into a SpodeianLakeDataset.
        """
        lake_path = Path(lake_dir) if lake_dir else get_default_lake_path()
        registry_data = load_lake_registry(lake_path)
        partitions_dict = registry_data.get("partitions", {})

        # Collect matching partitions
        matching = []
        for key, meta in partitions_dict.items():
            ds_id = meta.get("dataset_id")
            if ds_id == dataset_id or key.startswith(f"{dataset_id}:"):
                matching.append(meta)

        if not matching:
            raise KeyError(
                f"Dataset ID '{dataset_id}' not found in LakeRegistry at {lake_path}. "
                f"Available datasets: {list({m.get('dataset_id') for m in partitions_dict.values()})}"
            )

        # Sort chronologically by created_at_epoch_s, then partition_key
        matching.sort(
            key=lambda m: (m.get("created_at_epoch_s", 0), str(m.get("partition_key", "")))
        )

        verified_files = []
        for meta in matching:
            fpath_str = meta.get("file_path", "")
            fpath = Path(fpath_str)
            if not fpath.is_absolute():
                fpath = lake_path / fpath
            if not fpath.exists():
                # Fallback to lake_path / data / fpath.name
                fallback = lake_path / "data" / fpath.name
                if fallback.exists():
                    fpath = fallback
                else:
                    raise FileNotFoundError(f"Partition file missing on disk: {fpath}")

            if verify_sha256:
                expected_sha = meta.get("sha256_checksum")
                if expected_sha:
                    file_bytes = fpath.read_bytes()
                    actual_sha = hashlib.sha256(file_bytes).hexdigest()
                    if actual_sha != expected_sha:
                        raise ValueError(
                            f"Integrity check failed for partition '{meta.get('partition_key')}': "
                            f"expected SHA-256 {expected_sha}, got {actual_sha}"
                        )

            verified_files.append(fpath)

        return cls(
            parquet_path=verified_files,
            feature_columns=feature_columns,
            partition_metadata=matching,
        )


def get_default_lake_path() -> Path:
    """Resolves default Windows Data Lake directory (%LOCALAPPDATA%/spodeian/lake)."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "spodeian" / "lake"
    return Path(".spodeian") / "lake"


def load_lake_registry(lake_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Reads the metadata registry of the local Windows Data Lake."""
    path = Path(lake_dir) if lake_dir else get_default_lake_path()
    registry_file = path / "lake_registry.json"
    if not registry_file.exists():
        return {"partitions": {}, "experiments": {}}
    with open(registry_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_synthetic_dataloader(dim: int = 16, samples: int = 256, batch_size: int = 32) -> DataLoader:
    """Creates a deterministic synthetic dataloader for interactive demonstration and testing."""
    torch.manual_seed(42)
    x = torch.randn(samples, dim)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)


def build_dataloader(
    data_path: Optional[Union[str, Path]] = None,
    lake_dataset: Optional[str] = None,
    lake_dir: Optional[Union[str, Path]] = None,
    feature_columns: Optional[List[str]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    dim: int = 16,
    samples: int = 256,
    verify_sha256: bool = True,
) -> DataLoader:
    """
    DataLakeConfig-aware DataLoader factory.
    Constructs a PyTorch DataLoader from real Parquet partitions (via path or lake dataset ID)
    or falls back cleanly to synthetic data when neither is specified.
    """
    if lake_dataset:
        dataset = SpodeianLakeDataset.from_registry(
            dataset_id=lake_dataset,
            lake_dir=lake_dir,
            feature_columns=feature_columns,
            verify_sha256=verify_sha256,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif data_path:
        dataset = SpodeianLakeDataset(parquet_path=data_path, feature_columns=feature_columns)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        return build_synthetic_dataloader(dim=dim, samples=samples, batch_size=batch_size)


def build_train_val_dataloaders(
    data_path: Optional[Union[str, Path]] = None,
    lake_dataset: Optional[str] = None,
    lake_dir: Optional[Union[str, Path]] = None,
    feature_columns: Optional[List[str]] = None,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    dim: int = 16,
    samples: int = 256,
    val_samples: int = 64,
    verify_sha256: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Constructs paired train and validation DataLoaders.
    Returns (train_loader, val_loader, feature_dim).
    """
    if lake_dataset:
        dataset = SpodeianLakeDataset.from_registry(
            dataset_id=lake_dataset,
            lake_dir=lake_dir,
            feature_columns=feature_columns,
            verify_sha256=verify_sha256,
        )
    elif data_path:
        dataset = SpodeianLakeDataset(parquet_path=data_path, feature_columns=feature_columns)
    else:
        train_loader = build_synthetic_dataloader(dim=dim, samples=samples, batch_size=batch_size)
        val_loader = build_synthetic_dataloader(dim=dim, samples=val_samples, batch_size=batch_size)
        return train_loader, val_loader, dim

    feature_dim = dataset.features.shape[1] if hasattr(dataset, "features") else dim
    total_len = len(dataset)
    if total_len < 4:
        # For very small datasets, reuse for train and val
        train_loader = DataLoader(dataset, batch_size=min(batch_size, total_len), shuffle=True)
        val_loader = DataLoader(dataset, batch_size=min(batch_size, total_len), shuffle=False)
    else:
        val_size = max(1, int(total_len * val_ratio))
        train_size = total_len - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, feature_dim
