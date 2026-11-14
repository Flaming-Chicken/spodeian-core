"""Standardized high-resolution micro-benchmarking, memory profiling, and FLOP counting."""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # type: ignore


@dataclass
class BenchmarkResult:
    name: str
    median_time_ms: float
    p90_time_ms: float
    p99_time_ms: float
    stdev_time_ms: float
    peak_ram_mb: float
    peak_vram_mb: float
    flops: int
    throughput_samples_per_sec: float = 0.0
    arithmetic_intensity: float = 0.0

    def as_row(self) -> list[Any]:
        return [
            self.name,
            f"{self.median_time_ms:.3f} ms (p90: {self.p90_time_ms:.3f})",
            f"{self.peak_ram_mb:.2f} MB",
            f"{self.peak_vram_mb:.2f} MB",
            f"{self.flops:,}" if self.flops > 0 else "N/A",
            (
                f"{self.throughput_samples_per_sec:,.1f}/s"
                if self.throughput_samples_per_sec > 0
                else "N/A"
            ),
        ]


def benchmark_fn(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    iters: int = 10,
    batch_size: int = 1,
    name: str = "Operation",
    **kwargs: Any,
) -> BenchmarkResult:
    """Accurately benchmarks latency (p50/p90/p99), peak RSS RAM, VRAM, and FLOP count of a callable.

    Args:
        fn: Function to benchmark.
        args: Positional arguments passed to fn.
        warmup: Number of untimed warmup iterations.
        iters: Number of timed iterations.
        batch_size: Batch size of input to compute samples/sec throughput.
        name: Name descriptor for the benchmark row.
        kwargs: Keyword arguments passed to fn.
    """
    gc.collect()
    is_cuda = torch.cuda.is_available() and any(
        isinstance(a, torch.Tensor) and a.is_cuda for a in args
    )

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize()

    # Track CPU RAM allocations
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    if hasattr(tracemalloc, "reset_peak"):
        tracemalloc.reset_peak()

    # Latency timing
    durations: list[float] = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        _ = fn(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter_ns()
        durations.append((end - start) / 1_000_000.0)  # ms

    current_mem, peak_ram_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()
    peak_ram_mb = peak_ram_bytes / (1024.0 * 1024.0)

    durations.sort()
    median_ms = statistics.median(durations)
    idx_p90 = min(int(len(durations) * 0.9), len(durations) - 1)
    idx_p99 = min(int(len(durations) * 0.99), len(durations) - 1)
    p90_ms = durations[idx_p90]
    p99_ms = durations[idx_p99]
    stdev_ms = statistics.stdev(durations) if len(durations) > 1 else 0.0

    throughput = (batch_size / (median_ms / 1000.0)) if median_ms > 0 else 0.0

    # VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if is_cuda else 0.0

    # Estimate FLOPs via PyTorch FlopCounterMode if available
    flops = 0
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as flop_counter:
            _ = fn(*args, **kwargs)
            flops = flop_counter.get_total_flops()
    except Exception:
        flops = 0

    return BenchmarkResult(
        name=name,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p99_time_ms=p99_ms,
        stdev_time_ms=stdev_ms,
        peak_ram_mb=peak_ram_mb,
        peak_vram_mb=peak_vram_mb,
        flops=flops,
        throughput_samples_per_sec=throughput,
    )


def compare_tracks(
    track_fns: dict[str, Callable[[], Any]],
    warmup: int = 3,
    iters: int = 10,
    batch_size: int = 1,
) -> list[BenchmarkResult]:
    """Runs benchmark across multiple track implementations (e.g. eager, compile, jax)."""
    results: list[BenchmarkResult] = []
    for name, fn in track_fns.items():
        res = benchmark_fn(fn, warmup=warmup, iters=iters, batch_size=batch_size, name=name)
        results.append(res)
    return results


def format_benchmark_table(results: list[BenchmarkResult]) -> str:
    """Formats benchmark results as a clean Markdown table."""
    headers = [
        "Implementation / Track",
        "Latency (ms)",
        "Peak RAM (MB)",
        "Peak VRAM (MB)",
        "FLOPs",
        "Throughput",
    ]
    rows = [r.as_row() for r in results]
    if tabulate is not None:
        return str(tabulate(rows, headers=headers, tablefmt="github"))

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def save_benchmark_baseline(
    results: list[BenchmarkResult], path: str | Path
) -> None:
    """Saves benchmark results as a JSON baseline file for regression testing."""
    import json
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        r.name: {
            "median_time_ms": r.median_time_ms,
            "p90_time_ms": r.p90_time_ms,
            "throughput": r.throughput_samples_per_sec,
            "flops": r.flops,
        }
        for r in results
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def check_benchmark_regression(
    results: list[BenchmarkResult], baseline_path: str | Path, tolerance_pct: float = 50.0
) -> list[str]:
    """Compares current benchmark results against a saved baseline file, flagging regressions."""
    import json
    from pathlib import Path

    b_path = Path(baseline_path)
    if not b_path.is_file():
        return []

    with open(b_path, encoding="utf-8") as f:
        baseline = json.load(f)

    regressions = []
    for r in results:
        if r.name in baseline:
            base_median = baseline[r.name]["median_time_ms"]
            if base_median > 0:
                slowdown_pct = ((r.median_time_ms - base_median) / base_median) * 100.0
                if slowdown_pct > tolerance_pct:
                    regressions.append(
                        f"Regression in '{r.name}': {r.median_time_ms:.3f}ms vs baseline {base_median:.3f}ms "
                        f"(+{slowdown_pct:.1f}% exceeds tolerance {tolerance_pct}%)"
                    )
    return regressions

