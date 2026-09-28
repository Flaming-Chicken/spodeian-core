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


def quantize_weights(
    tensor: torch.Tensor,
    precision: str = "float32",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Universal precision dispatcher.
    Supported precisions: "float32", "float16", "bfloat16", "int8", "posit16", "posit8", "ternary".
    """
    p = precision.lower().strip()
    if p in ("float32", "fp32"):
        return tensor.float(), {"format": "float32", "bits": 32, "compression_ratio": 1.0, "sqnr_db": 100.0}
    elif p in ("float16", "fp16", "half"):
        return quantize_float16(tensor)
    elif p in ("bfloat16", "bf16", "brain_float"):
        return quantize_bfloat16(tensor)
    elif p in ("int8", "qint8"):
        q_tensor, scale, meta = quantize_int8_symmetric(tensor)
        return q_tensor, meta
    elif p in ("posit16", "posit_16"):
        return quantize_posit(tensor, total_bits=16, es=1)
    elif p in ("posit8", "posit_8"):
        return quantize_posit(tensor, total_bits=8, es=0)
    elif p in ("ternary", "1.58b", "bitnet"):
        t_tensor, scale, meta = quantize_ternary_1_58b(tensor)
        return t_tensor, meta
    else:
        raise ValueError(f"Unsupported quantization precision: '{precision}'. Supported: float32, float16, bfloat16, int8, posit16, posit8, ternary")
