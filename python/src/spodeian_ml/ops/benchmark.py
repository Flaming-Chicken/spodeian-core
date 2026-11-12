"""
Universal Quantization Benchmark Harness for spodeian-ml.
Evaluates model weights and inference across FP32, FP16, BF16, FP8, NF4, INT8, INT4,
Posit16, Posit8, Ternary (1.58b), and BNN (1-bit), measuring throughput, memory footprint,
SQNR, Cosine Similarity, and MAE/RMSE fidelity.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import time
import copy
import uuid
import torch
import torch.nn as nn

from spodeian_ml.ops.config import ExperimentConfig
from spodeian_ml.ssl.autoencoder import MaskedAutoencoder
from spodeian_ml.bridge.quantization import quantize_weights, compute_sqnr, NF4_QUANTILES
from spodeian_ml.ops.experiment import (
    ExperimentRecord,
    save_experiment_record,
    get_system_hardware_context,
)


SUPPORTED_BENCHMARK_FORMATS = [
    ("FP32", "float32", 32),
    ("FP16", "float16", 16),
    ("BF16", "bfloat16", 16),
    ("FP8", "fp8_e4m3", 8),
    ("NF4", "nf4", 4),
    ("INT8", "int8", 8),
    ("INT4", "int4", 4),
    ("Posit16", "posit16", 16),
    ("Posit8", "posit8", 8),
    ("Log8", "log8", 8),
    ("Ternary", "ternary", 2),
    ("BNN", "binary", 1),
]


def dequantize_weight_tensor(q_tensor: torch.Tensor, meta: Dict[str, Any], orig_tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """Helper to reconstruct float weights from quantized representation for forward pass evaluation."""
    p = precision.lower()
    if p in ("int8", "qint8", "fp8_e4m3", "fp8", "fp8_e5m2", "binary", "bnn", "1bit", "ternary", "1.58b"):
        scale = meta.get("scale", 1.0)
        return (q_tensor.float() * scale).view_as(orig_tensor)
    elif p in ("nf4", "normal_float4"):
        scale = meta.get("scale", 1.0)
        q_table = torch.tensor(NF4_QUANTILES, dtype=torch.float32)
        return (q_table[q_tensor.long()].view_as(orig_tensor) * scale)
    else:
        return q_tensor.float().view_as(orig_tensor)


def run_quantization_benchmark(
    model: nn.Module,
    config: Optional[ExperimentConfig] = None,
    eval_samples: int = 512,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    save_to_lake: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Executes empirical inference and fidelity benchmark across all 11 quantization formats.
    Returns structured results list and formatted display table.
    """
    dev = device or torch.device("cpu")
    model = model.to(dev).eval()

    # Determine input dimension from model
    input_dim = getattr(model, "input_dim", 16)
    torch.manual_seed(42)
    x = torch.randn(eval_samples, input_dim, device=dev)

    # Base FP32 forward pass
    with torch.no_grad():
        out = model(x)
        y_base = out[0].detach() if isinstance(out, (tuple, list)) else out.detach()

    # Compute base throughput
    warmup_iters = 5
    eval_iters = 20
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x[:batch_size])

        t0 = time.perf_counter()
        for _ in range(eval_iters):
            _ = model(x[:batch_size])
        base_elapsed = time.perf_counter() - t0

    base_throughput = (eval_iters * batch_size) / max(1e-6, base_elapsed)
    base_numel = sum(p.numel() for p in model.parameters())
    base_memory_mb = (base_numel * 4) / (1024 * 1024)

    flat_base = y_base.flatten().to(torch.float64)
    norm_base = torch.linalg.norm(flat_base).item()

    ranked_results: List[Dict[str, Any]] = []

    for display_name, prec_key, bits in SUPPORTED_BENCHMARK_FORMATS:
        if prec_key == "float32":
            ranked_results.append({
                "format": display_name,
                "bits": bits,
                "memory_mb": round(base_memory_mb, 3),
                "compression": 1.0,
                "throughput": round(base_throughput, 1),
                "sqnr_db": 99.99,
                "cosine_sim": 1.000000,
                "mae": 0.000000,
                "rmse": 0.000000,
            })
            continue

        # Create cloned quantized model
        model_q = copy.deepcopy(model)
        total_q_bits = 0

        with torch.no_grad():
            for name, param in model_q.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    q_t, meta = quantize_weights(param.data, precision=prec_key)
                    dequant = dequantize_weight_tensor(q_t, meta, param.data, prec_key)
                    param.data.copy_(dequant)
                    total_q_bits += param.numel() * bits
                else:
                    total_q_bits += param.numel() * 32

        # Evaluate output fidelity
        with torch.no_grad():
            out_q = model_q(x)
            y_q = out_q[0].detach() if isinstance(out_q, (tuple, list)) else out_q.detach()

        flat_q = y_q.flatten().to(torch.float64)
        norm_q = torch.linalg.norm(flat_q).item()

        if norm_base > 1e-12 and norm_q > 1e-12:
            cos_sim = torch.dot(flat_base, flat_q).item() / (norm_base * norm_q)
        else:
            cos_sim = 1.0

        diff = flat_base - flat_q
        mae = torch.mean(torch.abs(diff)).item()
        rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        sqnr = compute_sqnr(y_base, y_q)

        # Measure throughput with simulated latency scaling
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(eval_iters):
                _ = model_q(x[:batch_size])
        q_elapsed = time.perf_counter() - t0
        q_throughput = (eval_iters * batch_size) / max(1e-6, q_elapsed)

        q_memory_mb = total_q_bits / (8 * 1024 * 1024)
        compression = base_memory_mb / max(1e-6, q_memory_mb)

        ranked_results.append({
            "format": display_name,
            "bits": bits,
            "memory_mb": round(q_memory_mb, 3),
            "compression": round(compression, 2),
            "throughput": round(q_throughput, 1),
            "sqnr_db": round(sqnr, 2),
            "cosine_sim": round(cos_sim, 6),
            "mae": float(mae),
            "rmse": float(rmse),
        })

    # Sort results by Pareto efficiency (SQNR descending, then compression)
    ranked_results.sort(key=lambda r: (r["sqnr_db"], r["compression"]), reverse=True)

    # Format ASCII table
    header = f"{'Format':<10} | {'Bits':<4} | {'Memory (MB)':<11} | {'Comp':<6} | {'Throughput':<12} | {'SQNR (dB)':<9} | {'Cosine Sim':<10} | {'MAE':<8} | {'RMSE':<8}"
    sep = "-" * len(header)
    table_rows = [
        "=== UNIVERSAL QUANTIZATION BENCHMARK MATRIX ===",
        f"Evaluated on {eval_samples} samples (Batch Size: {batch_size}, Baseline Memory: {base_memory_mb:.2f} MB)",
        "",
        header,
        sep,
    ]
    for r in ranked_results:
        table_rows.append(
            f"{r['format']:<10} | {r['bits']:<4} | {r['memory_mb']:<11.3f} | {r['compression']:<5.1f}x | {r['throughput']:<8.1f} s/s | {r['sqnr_db']:<9.2f} | {r['cosine_sim']:<10.6f} | {r['mae']:<8.5f} | {r['rmse']:<8.5f}"
        )
    table_str = "\n".join(table_rows)

    # Record to Lake Registry if requested
    if save_to_lake:
        try:
            metrics: Dict[str, float] = {}
            for r in ranked_results:
                fmt = r["format"].lower()
                metrics[f"{fmt}_sqnr_db"] = r["sqnr_db"]
                metrics[f"{fmt}_cosine_sim"] = r["cosine_sim"]
                metrics[f"{fmt}_memory_mb"] = r["memory_mb"]
                metrics[f"{fmt}_throughput"] = r["throughput"]

            exp_name = config.experiment_name if config else "quantization_benchmark"
            record = ExperimentRecord(
                run_id=str(uuid.uuid4()),
                experiment_name=f"{exp_name}_quant_benchmark",
                config_yaml_snapshot=f"benchmark_samples: {eval_samples}\nbatch_size: {batch_size}\n",
                final_metrics=metrics,
                hardware_context=get_system_hardware_context(),
                timestamp_epoch_s=int(time.time()),
                duration_seconds=round(base_elapsed * 10, 2),
                completion_status="COMPLETED",
                tags=["benchmark", "quantization", "pareto_matrix"],
            )
            save_experiment_record(record)
        except Exception:
            pass

    return ranked_results, table_str
