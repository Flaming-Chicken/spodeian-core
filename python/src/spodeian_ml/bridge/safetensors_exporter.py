"""
Safetensors Exporter Bridge.
Extracts inference-ready model weights from PyTorch, strips offline training artifacts,
and serializes pure .safetensors files matching Candle VarBuilder naming paths.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch
import torch.nn as nn
from safetensors.torch import save_file


def export_encoder_to_safetensors(
    model: nn.Module,
    output_path: Path,
    prefix_filter: Optional[str] = "encoder",
    manifest_path: Optional[Path] = None,
    dtype: str = "float32",
) -> Path:
    """
    Exports encoder weights to a clean .safetensors file for zero-overhead loading in Candle.
    Drops optimizer state, reconstruction decoders, and non-inference tensors.
    Supports "float32" and "float16" half-precision quantization.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    exported_tensors: Dict[str, torch.Tensor] = {}
    manifest_entries: Dict[str, Any] = {}

    for key, tensor in state_dict.items():
        # Only export target prefix if specified (e.g., "encoder")
        if prefix_filter and not key.startswith(prefix_filter):
            continue

        clean_tensor = tensor.detach().cpu().contiguous()
        if dtype.lower() == "float16":
            clean_tensor = clean_tensor.half()
        elif dtype.lower() == "float32":
            clean_tensor = clean_tensor.float()

        exported_tensors[key] = clean_tensor

        manifest_entries[key] = {
            "shape": list(clean_tensor.shape),
            "dtype": str(clean_tensor.dtype),
            "numel": clean_tensor.numel(),
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
                "total_tensors": len(exported_tensors),
                "tensors": manifest_entries,
            }, f, indent=2)

    return output_path
