"""Core package for AI-Examples: deterministic micro-generators, metrics, benchmarks, parity, and export."""

from __future__ import annotations

from . import benchmark, data, export, metrics, parity
from .env import EnvInfo, setup_env

__all__ = [
    "EnvInfo",
    "setup_env",
    "data",
    "metrics",
    "benchmark",
    "parity",
    "export",
]
