"""
Safetensors Exporter Bridge.
Extracts inference-ready model weights from PyTorch, strips offline training artifacts,
and serializes pure .safetensors files with universal quantization support
(Float32, Float16, Brain Float BFloat16, INT8, and Posit emulation).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch
import torch.nn as nn
from safetensors.torch import save_file

from spodeian_ml.bridge.quantization import quantize_weights


def export_encoder_to_safetensors(
    model: nn.Module,
    output_path: Path,
    prefix_filter: Optional[str] = "encoder",
    manifest_path: Optional[Path] = None,
    precision: str = "float32",
) -> Path:
    """
    Exports encoder weights to a clean .safetensors file matching Candle VarBuilder naming paths.
    Supports "float32", "float16", "bfloat16", "int8", "posit16", and "posit8".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    exported_tensors: Dict[str, torch.Tensor] = {}
    manifest_entries: Dict[str, Any] = {}

    for key, tensor in state_dict.items():
        if prefix_filter and not key.startswith(prefix_filter):
            continue

        cpu_tensor = tensor.detach().cpu().contiguous()
        q_tensor, meta = quantize_weights(cpu_tensor, precision=precision)

        exported_tensors[key] = q_tensor

        # For INT8 quantization, export the scale factor alongside the weights
        if meta.get("format") == "int8_symmetric" and "scale" in meta:
            scale_tensor = torch.tensor([meta["scale"]], dtype=torch.float32)
            exported_tensors[f"{key}__scale"] = scale_tensor

        manifest_entries[key] = {
            "shape": list(cpu_tensor.shape),
            "original_dtype": str(cpu_tensor.dtype),
            "quantized_dtype": str(q_tensor.dtype),
            "numel": cpu_tensor.numel(),
            "quantization": meta,
        }

    # Save to pure safetensors
    save_file(exported_tensors, str(output_path))

    # Optional schema manifest JSON for Rust type/dimension verification
    if manifest_path:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "safetensor_file": output_path.name,
                "precision": precision,
                "total_tensors": len(exported_tensors),
                "tensors": manifest_entries,
            }, f, indent=2)

    return output_path
