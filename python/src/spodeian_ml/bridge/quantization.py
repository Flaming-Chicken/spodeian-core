"""
Universal Numerical Precision & Quantization Engine for spodeian-ml.
Supports IEEE Float32, Float16, Google Brain Float (BFloat16), INT8 Linear Scaling,
and Type III Unum Posit (Posit8, Posit16) dynamic range tapered quantization.
"""

from typing import Tuple, Dict, Any, Union
import math
import torch
import torch.nn as nn


def compute_sqnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Computes Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB)."""
    orig_f64 = original.to(torch.float64)
    quant_f64 = quantized.to(torch.float64)
    signal_power = torch.mean(orig_f64 ** 2).item()
    noise_power = torch.mean((orig_f64 - quant_f64) ** 2).item()
    if noise_power < 1e-12:
        return 100.0  # Perfect reconstruction
    if signal_power < 1e-12:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def quantize_bfloat16(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Quantizes to Google Brain Float 16 (BFloat16: 1 sign, 8 exponent, 7 mantissa)."""
    bf16_tensor = tensor.to(torch.bfloat16)
    sqnr = compute_sqnr(tensor, bf16_tensor.float())
    meta = {
        "format": "bfloat16",
        "bits": 16,
        "compression_ratio": 2.0,
        "sqnr_db": round(sqnr, 2),
    }
    return bf16_tensor, meta


def quantize_float16(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Quantizes to IEEE 754 Half Precision (Float16: 1 sign, 5 exponent, 10 mantissa)."""
    fp16_tensor = tensor.to(torch.float16)
    sqnr = compute_sqnr(tensor, fp16_tensor.float())
    meta = {
        "format": "float16",
        "bits": 16,
        "compression_ratio": 2.0,
        "sqnr_db": round(sqnr, 2),
    }
    return fp16_tensor, meta


def quantize_int8_symmetric(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Quantizes to 8-bit signed integer with symmetric scaling:
    scale = max(|tensor|) / 127.0
    quantized = clamp(round(tensor / scale), -128, 127)
    """
    max_val = torch.max(torch.abs(tensor)).item()
    scale = max(max_val / 127.0, 1e-8)
    int8_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)

    dequantized = int8_tensor.float() * scale
    sqnr = compute_sqnr(tensor, dequantized)

    meta = {
        "format": "int8_symmetric",
        "bits": 8,
        "scale": scale,
        "compression_ratio": 4.0,
        "sqnr_db": round(sqnr, 2),
    }
    return int8_tensor, scale, meta


def emulate_posit_rounding(tensor: torch.Tensor, total_bits: int = 16, es: int = 1) -> torch.Tensor:
    """
    Mathematical emulation of Type III Unum Posit arithmetic (John Gustafson).
    Posits feature 'tapered precision': numbers near 1.0 possess maximal fraction bits,
    with dynamic range gracefully trading off mantissa precision at extreme values.
    """
    flat = tensor.flatten()
    signs = torch.sign(flat)
    abs_vals = torch.abs(flat)

    useed = 2 ** (2 ** es)
    max_val = useed ** (total_bits - 2)
    min_val = useed ** -(total_bits - 2)

    clamped = torch.clamp(abs_vals, min=min_val, max=max_val)

    # Compute regime k
    log_useed = math.log2(useed)
    log_vals = torch.log2(clamped) / log_useed
    k = torch.floor(log_vals)

    # Remaining bits for exponent and fraction
    # In Posit: regime takes |k| + 2 bits (or k + 1 for k >= 0)
    regime_len = torch.where(k >= 0, k + 2, -k + 1)
    avail_bits = torch.clamp(total_bits - 1 - regime_len - es, min=1, max=total_bits - 3)

    # Quantize fraction according to available tapered bits
    frac_levels = 2.0 ** avail_bits
    scale = (2.0 ** (k * (2 ** es)))
    mantissa = clamped / torch.clamp(scale, min=1e-12)
    quantized_mantissa = torch.round(mantissa * frac_levels) / frac_levels

    reconstructed = signs * quantized_mantissa * scale
    # Zero stays zero
    reconstructed = torch.where(flat == 0.0, torch.zeros_like(reconstructed), reconstructed)
    return reconstructed.view_as(tensor)


def quantize_posit(tensor: torch.Tensor, total_bits: int = 16, es: int = 1) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Quantizes tensor according to Posit(total_bits, es) precision.
    Default: Posit16 (16-bit, es=1) or Posit8 (8-bit, es=0).
    """
    posit_tensor = emulate_posit_rounding(tensor, total_bits=total_bits, es=es)
    sqnr = compute_sqnr(tensor, posit_tensor)

    meta = {
        "format": f"posit{total_bits}",
        "total_bits": total_bits,
        "exponent_size": es,
        "compression_ratio": 32.0 / total_bits,
        "sqnr_db": round(sqnr, 2),
    }
    return posit_tensor, meta


def quantize_ternary_1_58b(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Quantizes to balanced ternary (1.58-bit: -1, 0, 1) following the BitNet 1.58b specification.
    scale = mean(|tensor|)
    quantized = clamp(round(tensor / scale), -1, 1)
    """
    scale = torch.mean(torch.abs(tensor)).item()
    if scale < 1e-8:
        scale = 1e-8

    ternary_tensor = torch.clamp(torch.round(tensor / scale), -1, 1).to(torch.int8)

    dequantized = ternary_tensor.float() * scale
    sqnr = compute_sqnr(tensor, dequantized)

    meta = {
        "format": "ternary_1.58b",
        "bits": 1.58,
        "scale": scale,
        "compression_ratio": 32.0 / 1.58,
        "sqnr_db": round(sqnr, 2),
    }
    return ternary_tensor, scale, meta


def quantize_fp8_e4m3(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Quantizes to FP8 E4M3 (1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits).
    Max representable value: 448.0.
    """
    max_val = torch.max(torch.abs(tensor)).item()
    scale = max(max_val / 448.0, 1e-8)
    scaled = tensor / scale

    # Check if native torch float8_e4m3fn exists
    if hasattr(torch, "float8_e4m3fn"):
        fp8_tensor = scaled.to(torch.float8_e4m3fn)
        dequantized = fp8_tensor.float() * scale
    else:
        # Mathematical emulation
        clamped = torch.clamp(scaled, -448.0, 448.0)
        signs = torch.sign(clamped)
        abs_vals = torch.abs(clamped)
        nonzero = abs_vals > 1e-6
        # 3 bits of mantissa = 8 levels between powers of 2
        log2_vals = torch.floor(torch.log2(torch.clamp(abs_vals, min=1e-6)))
        exp_scale = 2.0 ** log2_vals
        mantissa = (abs_vals / exp_scale - 1.0)
        quant_mantissa = torch.round(mantissa * 8.0) / 8.0
        dequantized = signs * (1.0 + quant_mantissa) * exp_scale
        dequantized = torch.where(nonzero, dequantized, torch.zeros_like(dequantized)) * scale
        fp8_tensor = dequantized.half()

    sqnr = compute_sqnr(tensor, dequantized)
    meta = {
        "format": "fp8_e4m3",
        "bits": 8,
        "scale": scale,
        "compression_ratio": 4.0,
        "sqnr_db": round(sqnr, 2),
    }
    return fp8_tensor, scale, meta


def quantize_fp8_e5m2(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Quantizes to FP8 E5M2 (1 sign bit, 5 exponent bits with bias 15, 2 mantissa bits).
    Max representable value: 57344.0.
    """
    max_val = torch.max(torch.abs(tensor)).item()
    scale = max(max_val / 57344.0, 1e-8)
    scaled = tensor / scale

    if hasattr(torch, "float8_e5m2"):
        fp8_tensor = scaled.to(torch.float8_e5m2)
        dequantized = fp8_tensor.float() * scale
    else:
        clamped = torch.clamp(scaled, -57344.0, 57344.0)
        signs = torch.sign(clamped)
        abs_vals = torch.abs(clamped)
        nonzero = abs_vals > 1e-6
        log2_vals = torch.floor(torch.log2(torch.clamp(abs_vals, min=1e-6)))
        exp_scale = 2.0 ** log2_vals
        mantissa = (abs_vals / exp_scale - 1.0)
        quant_mantissa = torch.round(mantissa * 4.0) / 4.0
        dequantized = signs * (1.0 + quant_mantissa) * exp_scale
        dequantized = torch.where(nonzero, dequantized, torch.zeros_like(dequantized)) * scale
        fp8_tensor = dequantized.half()

    sqnr = compute_sqnr(tensor, dequantized)
    meta = {
        "format": "fp8_e5m2",
        "bits": 8,
        "scale": scale,
        "compression_ratio": 4.0,
        "sqnr_db": round(sqnr, 2),
    }
    return fp8_tensor, scale, meta


# Standard 16 NormalFloat4 quantiles for N(0, 1) Gaussian distributions
NF4_QUANTILES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]


def quantize_nf4(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Quantizes to NormalFloat4 (NF4: 4-bit information-theoretically optimal for Gaussian weights).
    """
    max_val = torch.max(torch.abs(tensor)).item()
    scale = max(max_val, 1e-8)
    normalized = tensor / scale

    q_table = torch.tensor(NF4_QUANTILES, dtype=tensor.dtype, device=tensor.device)
    flat_norm = normalized.flatten().unsqueeze(1)
    # Find nearest quantile index
    dists = torch.abs(flat_norm - q_table.unsqueeze(0))
    indices = torch.argmin(dists, dim=1).to(torch.int8)

    dequantized = q_table[indices.long()].view_as(tensor) * scale
    sqnr = compute_sqnr(tensor, dequantized)

    meta = {
        "format": "nf4",
        "bits": 4,
        "scale": scale,
        "compression_ratio": 8.0,
        "sqnr_db": round(sqnr, 2),
    }
    return indices.view_as(tensor), scale, meta


def quantize_int4_groupwise(tensor: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Quantizes to 4-bit integer [-8, 7] with group-wise scaling (AWQ/GPTQ style).
    """
    orig_shape = tensor.shape
    flat = tensor.flatten()
    pad_len = (group_size - (flat.numel() % group_size)) % group_size
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))

    groups = flat.view(-1, group_size)
    max_vals = torch.max(torch.abs(groups), dim=1, keepdim=True).values
    scales = torch.clamp(max_vals / 7.0, min=1e-8)
    q_groups = torch.clamp(torch.round(groups / scales), -8, 7).to(torch.int8)

    dequant_groups = q_groups.float() * scales
    dequant = dequant_groups.flatten()[:tensor.numel()].view_as(tensor)
    sqnr = compute_sqnr(tensor, dequant)

    meta = {
        "format": "int4_groupwise",
        "bits": 4,
        "group_size": group_size,
        "compression_ratio": 8.0,
        "sqnr_db": round(sqnr, 2),
    }
    return q_groups.view_as(tensor if pad_len == 0 else flat[:tensor.numel()].view_as(tensor)), scales.squeeze(), meta


def quantize_binary_1bit(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Quantizes to pure 1-bit binary weights (+1, -1) following Binary Neural Network (BNN) specification.
    scale = mean(|tensor|)
    quantized = where(tensor >= 0, 1, -1)
    """
    scale = torch.mean(torch.abs(tensor)).item()
    if scale < 1e-8:
        scale = 1e-8

    binary_tensor = torch.where(tensor >= 0, torch.tensor(1, dtype=torch.int8), torch.tensor(-1, dtype=torch.int8))
    dequantized = binary_tensor.float() * scale
    sqnr = compute_sqnr(tensor, dequantized)

    meta = {
        "format": "binary_1bit",
        "bits": 1.0,
        "scale": scale,
        "compression_ratio": 32.0,
        "sqnr_db": round(sqnr, 2),
    }
    return binary_tensor, scale, meta


def quantize_logarithmic(tensor: torch.Tensor, base: float = 2.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Quantizes to base-2 logarithmic scaling: sign(x) * base^round(log_base(|x|)).
    Dedicated high resolution near zero, perfectly tailored for heavy-tailed power-law financial returns.
    """
    signs = torch.sign(tensor)
    abs_vals = torch.abs(tensor)
    nonzero = abs_vals > 1e-8

    log_base = math.log(base)
    log_vals = torch.log(torch.clamp(abs_vals, min=1e-8)) / log_base
    exponent = torch.clamp(torch.round(log_vals), min=-16.0, max=15.0)

    dequantized = signs * (base ** exponent)
    dequantized = torch.where(nonzero, dequantized, torch.zeros_like(dequantized))
    sqnr = compute_sqnr(tensor, dequantized)

    meta = {
        "format": "logarithmic",
        "bits": 5,  # 1 sign bit + 4 exponent bits [-16, 15]
        "base": base,
        "compression_ratio": 32.0 / 5.0,
        "sqnr_db": round(sqnr, 2),
    }
    return dequantized, meta


def quantize_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    precision: str = "int8",
) -> Tuple[Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any], Dict[str, Any]]:
    """
    Quantizes Key-Value memory cache for autoregressive inference, slashing VRAM bandwidth bottlenecks.
    Supports "int8" and "fp8".
    """
    if precision.lower() == "fp8":
        q_k, scale_k, meta_k = quantize_fp8_e4m3(key)
        q_v, scale_v, meta_v = quantize_fp8_e4m3(value)
    else:
        q_k, scale_k, meta_k = quantize_int8_symmetric(key)
        q_v, scale_v, meta_v = quantize_int8_symmetric(value)

    meta = {
        "format": f"kv_cache_{precision}",
        "key_meta": meta_k,
        "value_meta": meta_v,
        "compression_ratio": meta_k.get("compression_ratio", 4.0),
    }
    return (q_k, scale_k), (q_v, scale_v), meta


def quantize_weights(
    tensor: torch.Tensor,
    precision: str = "float32",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Universal precision dispatcher.
    Supported precisions:
      "float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2", "int8", "posit16", "posit8",
      "nf4", "int4", "binary", "ternary", "logarithmic".
    """
    p = precision.lower().strip()
    if p in ("float32", "fp32"):
        return tensor.float(), {"format": "float32", "bits": 32, "compression_ratio": 1.0, "sqnr_db": 100.0}
    elif p in ("float16", "fp16", "half"):
        return quantize_float16(tensor)
    elif p in ("bfloat16", "bf16", "brain_float"):
        return quantize_bfloat16(tensor)
    elif p in ("fp8_e4m3", "fp8", "float8"):
        q_tensor, scale, meta = quantize_fp8_e4m3(tensor)
        return q_tensor, meta
    elif p in ("fp8_e5m2",):
        q_tensor, scale, meta = quantize_fp8_e5m2(tensor)
        return q_tensor, meta
    elif p in ("int8", "qint8"):
        q_tensor, scale, meta = quantize_int8_symmetric(tensor)
        return q_tensor, meta
    elif p in ("nf4", "normal_float4"):
        q_tensor, scale, meta = quantize_nf4(tensor)
        return q_tensor, meta
    elif p in ("int4", "qint4", "int4_groupwise"):
        q_tensor, scales, meta = quantize_int4_groupwise(tensor)
        return q_tensor, meta
    elif p in ("binary", "bnn", "1bit"):
        q_tensor, scale, meta = quantize_binary_1bit(tensor)
        return q_tensor, meta
    elif p in ("posit16", "posit_16"):
        return quantize_posit(tensor, total_bits=16, es=1)
    elif p in ("posit8", "posit_8"):
        return quantize_posit(tensor, total_bits=8, es=0)
    elif p in ("ternary", "1.58b", "bitnet"):
        t_tensor, scale, meta = quantize_ternary_1_58b(tensor)
        return t_tensor, meta
    elif p in ("logarithmic", "log", "log2"):
        q_tensor, meta = quantize_logarithmic(tensor)
        return q_tensor, meta
    else:
        raise ValueError(
            f"Unsupported quantization precision: '{precision}'. "
            "Supported: float32, float16, bfloat16, fp8_e4m3, fp8_e5m2, int8, posit16, posit8, nf4, int4, binary, ternary, logarithmic"
        )


def benchmark_quantization_sqnr(
    tensor: torch.Tensor,
    precisions: Tuple[str, ...] = (
        "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2", "int8", "posit16", "posit8",
        "nf4", "int4", "binary", "ternary", "logarithmic"
    ),
) -> Dict[str, Any]:
    """
    Benchmarks Signal-to-Quantization-Noise Ratio (SQNR) and distortion across the complete Universal Quantization Precision Matrix.
    Returns structured metrics including SQNR (dB), Cosine Similarity, MAE, RMSE, and Compression Ratio.
    """
    flat_orig = tensor.detach().flatten().to(torch.float64)
    norm_orig = torch.linalg.norm(flat_orig).item()
    results = {}

    for prec in precisions:
        q_tensor, meta = quantize_weights(tensor, precision=prec)
        # Dequantize to float for fidelity metrics
        if prec.lower() in ("int8", "qint8", "fp8_e4m3", "fp8", "fp8_e5m2", "binary", "bnn", "1bit", "ternary", "1.58b", "bitnet"):
            scale = meta.get("scale", 1.0)
            dequant = q_tensor.float() * scale
        elif prec.lower() in ("nf4", "normal_float4"):
            scale = meta.get("scale", 1.0)
            q_table = torch.tensor(NF4_QUANTILES, dtype=torch.float32)
            dequant = q_table[q_tensor.long()].view_as(tensor) * scale
        elif prec.lower() in ("int4", "qint4", "int4_groupwise"):
            dequant = q_tensor.float()
        else:
            dequant = q_tensor.float()

        flat_dequant = dequant.flatten().to(torch.float64)
        norm_dequant = torch.linalg.norm(flat_dequant).item()

        # Cosine similarity
        if norm_orig > 1e-12 and norm_dequant > 1e-12:
            cos_sim = torch.dot(flat_orig, flat_dequant).item() / (norm_orig * norm_dequant)
        else:
            cos_sim = 1.0

        # MAE and RMSE
        diff = flat_orig - flat_dequant
        mae = torch.mean(torch.abs(diff)).item()
        rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        sqnr = compute_sqnr(tensor, dequant)

        results[prec] = {
            "format": meta.get("format", prec),
            "bits": meta.get("bits", meta.get("total_bits", 32)),
            "compression_ratio": meta.get("compression_ratio", 1.0),
            "sqnr_db": round(sqnr, 2),
            "cosine_similarity": round(cos_sim, 6),
            "mae": float(mae),
            "rmse": float(rmse),
        }

    return results

