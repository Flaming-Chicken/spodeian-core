"""
Spodeian ML Data Module: Augmentations, Factor Computations, and Data Lake Loaders.
"""

from spodeian_ml.data.lake_loader import (
    SpodeianLakeDataset,
    get_default_lake_path,
    load_lake_registry,
    build_dataloader,
    build_synthetic_dataloader,
    build_train_val_dataloaders,
)
from spodeian_ml.data.augmentations import (
    temporal_mask,
    gaussian_jitter,
    rolling_zscore,
    time_warp,
)
from spodeian_ml.data.factors import (
    log_returns,
    rolling_volatility,
    exponential_moving_average,
    vwap,
)

__all__ = [
    "SpodeianLakeDataset",
    "get_default_lake_path",
    "load_lake_registry",
    "build_dataloader",
    "build_synthetic_dataloader",
    "build_train_val_dataloaders",
    "temporal_mask",
    "gaussian_jitter",
    "rolling_zscore",
    "time_warp",
    "log_returns",
    "rolling_volatility",
    "exponential_moving_average",
    "vwap",
]
