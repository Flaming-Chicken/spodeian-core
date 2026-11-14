"""Universal safetensors checkpoint serialization and transitive numerical parity harness."""

from __future__ import annotations

from pathlib import Path

import torch

try:
    from safetensors.torch import load_file as _safetensors_load
    from safetensors.torch import save_file as _safetensors_save
except ImportError:
    _safetensors_load = None  # type: ignore
    _safetensors_save = None  # type: ignore


def save_reference_checkpoint(
    path: str | Path,
    weights_dict: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    metadata: dict[str, str] | None = None,
) -> None:
    """Saves model weights, test input, and golden output to a deterministic safetensors file.

    Args:
        path: File destination path (must end in .safetensors).
        weights_dict: Parameter dict mapping string names to tensors.
        input_tensor: Reference test input tensor.
        output_tensor: Reference golden output tensor from seed-42 run.
        metadata: Optional dictionary of string key-value metadata.
    """
    if _safetensors_save is None:
        raise ImportError("safetensors is required. Install via `pip install safetensors`.")

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    tensors_to_save: dict[str, torch.Tensor] = {}
    for k, v in weights_dict.items():
        # Standardize parameter keys with 'layers.' or direct names
        key = f"layers.{k}" if not k.startswith("layers.") else k
        tensors_to_save[key] = v.detach().contiguous().cpu()

    tensors_to_save["test_input"] = input_tensor.detach().contiguous().cpu()
    tensors_to_save["test_output"] = output_tensor.detach().contiguous().cpu()

    meta = {"format": "ai-examples-v1", "seed": "42"}
    if metadata:
        meta.update(metadata)

    _safetensors_save(tensors_to_save, str(target_path), metadata=meta)


def load_reference_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, str]]:
    """Loads a reference checkpoint returning weights, test input, golden output, and metadata.

    Args:
        path: Path to the .safetensors file.
        device: Device on which to place loaded tensors.

    Returns:
        tuple of (weights_dict, test_input, test_output, metadata_dict).
    """
    if _safetensors_load is None:
        raise ImportError("safetensors is required. Install via `pip install safetensors`.")

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    loaded: dict[str, torch.Tensor] = _safetensors_load(str(file_path), device=str(device))

    test_input = loaded.pop("test_input")
    test_output = loaded.pop("test_output")

    # Clean parameter keys
    weights_dict: dict[str, torch.Tensor] = {}
    for k, v in loaded.items():
        cleaned_key = k[len("layers.") :] if k.startswith("layers.") else k
        weights_dict[cleaned_key] = v

    # Extract metadata using safetensors safe_open if available
    metadata: dict[str, str] = {}
    try:
        from safetensors import safe_open

        with safe_open(str(file_path), framework="pt", device=str(device)) as f:
            meta = f.metadata()
            if meta:
                metadata = dict(meta)
    except Exception:
        pass

    return weights_dict, test_input, test_output, metadata


def verify_transitive_parity(
    outputs: dict[str, torch.Tensor],
    tol: float = 1e-5,
) -> tuple[bool, dict[tuple[str, str], float], str]:
    """Computes pairwise Chebyshev distance ||Y_i - Y_j||_inf across all framework outputs.

    Args:
        outputs: Mapping of framework names to output tensors (e.g. {'torch': t1, 'jax': t2}).
        tol: Maximum permissible absolute error (default 1e-5).

    Returns:
        tuple of (all_passed, pairwise_max_errors, formatted_summary_table).
    """
    names = sorted(outputs.keys())
    if len(names) < 2:
        return True, {}, "Single output provided; transitive comparison requires >= 2 frameworks."

    pairwise_errors: dict[tuple[str, str], float] = {}
    all_passed = True
    rows: list[str] = [
        "| Framework Pair | Max Abs Error (||Δ||_∞) | Mean Abs Error | Status |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            t_a = outputs[name_a].detach().cpu().to(dtype=torch.float64)
            t_b = outputs[name_b].detach().cpu().to(dtype=torch.float64)

            if t_a.shape != t_b.shape:
                all_passed = False
                err_str = f"Shape mismatch: {list(t_a.shape)} vs {list(t_b.shape)}"
                rows.append(f"| {name_a} ↔ {name_b} | {err_str} | N/A | ❌ FAIL |")
                pairwise_errors[(name_a, name_b)] = float("inf")
                continue

            max_err = float(torch.max(torch.abs(t_a - t_b)).item())
            mean_err = float(torch.mean(torch.abs(t_a - t_b)).item())
            pairwise_errors[(name_a, name_b)] = max_err

            passed = max_err <= tol
            if not passed:
                all_passed = False
            status = "✅ PASS" if passed else f"❌ FAIL (> {tol:g})"
            rows.append(f"| {name_a} ↔ {name_b} | {max_err:.3e} | {mean_err:.3e} | {status} |")

    summary_str = "\n".join(rows)
    return all_passed, pairwise_errors, summary_str
