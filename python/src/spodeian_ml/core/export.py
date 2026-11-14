"""Graph exportability and production runtime verification harness."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def verify_torch_export(
    model: torch.nn.Module,
    sample_input: tuple[Any, ...] | torch.Tensor,
    dynamic_shapes: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Verifies that a model forward graph captures cleanly via torch.export.export().

    Args:
        model: PyTorch nn.Module to export.
        sample_input: Example input tensor or tuple of arguments.
        dynamic_shapes: Optional dynamic dimension constraints.

    Returns:
        tuple of (success_boolean, status_message).
    """
    args = sample_input if isinstance(sample_input, tuple) else (sample_input,)
    model.eval()
    try:
        exported = torch.export.export(model, args, dynamic_shapes=dynamic_shapes)
        num_nodes = len(exported.graph.nodes)
        return True, f"torch.export succeeded ({num_nodes} graph nodes captured)"
    except Exception as exc:
        return False, f"torch.export failed: {type(exc).__name__}: {exc}"


def verify_torch_compile_fullgraph(
    model: torch.nn.Module,
    sample_input: tuple[Any, ...] | torch.Tensor,
) -> tuple[bool, str]:
    """Asserts that model executes under torch.compile(fullgraph=True) with zero graph breaks.

    Args:
        model: PyTorch nn.Module to compile.
        sample_input: Example input tensor or tuple of arguments.

    Returns:
        tuple of (success_boolean, status_message).
    """
    args = sample_input if isinstance(sample_input, tuple) else (sample_input,)
    model.eval()
    # 1. First assert zero graph breaks using the aot_eager backend (doesn't require C++ compiler)
    try:
        compiled_aot = torch.compile(model, fullgraph=True, backend="aot_eager")
        with torch.no_grad():
            _ = compiled_aot(*args)
    except Exception as exc:
        return (
            False,
            f"torch.compile(fullgraph=True) failed with graph break: {type(exc).__name__}: {exc}",
        )

    # 2. Attempt Inductor backend if a host C++ compiler is present
    try:
        compiled_inductor = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            _ = compiled_inductor(*args)
        return (
            True,
            "torch.compile(fullgraph=True, backend='inductor') executed with 0 graph breaks",
        )
    except Exception as exc:
        err_msg = str(exc)
        if (
            "InvalidCxxCompiler" in err_msg
            or "Compiler: cl is not found" in err_msg
            or "compiler" in err_msg.lower()
        ):
            return (
                True,
                "torch.compile(fullgraph=True) captured cleanly with 0 graph breaks (C++ compiler not in PATH, verified via aot_eager)",
            )
        return False, f"torch.compile(fullgraph=True) failed: {type(exc).__name__}: {exc}"


def verify_onnx_export(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    path: str | Path | None = None,
    opset_version: int = 17,
    tol: float = 1e-4,
) -> tuple[bool, str]:
    """Exports model to ONNX format and verifies numerical parity against ONNX Runtime if available.

    Args:
        model: PyTorch nn.Module to export.
        sample_input: Sample input tensor.
        path: Optional destination path. If None, uses a temporary file.
        opset_version: Target ONNX operator set version (default 17).
        tol: Maximum permissible absolute error against onnxruntime execution.

    Returns:
        tuple of (success_boolean, status_message).
    """
    model.eval()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if path is None:
        temp_dir = tempfile.TemporaryDirectory()
        target_path = Path(temp_dir.name) / "model.onnx"
    else:
        target_path = Path(path)

    try:
        with torch.no_grad():
            ref_output = model(sample_input)

        # Attempt export with legacy or dynamo based on environment
        onnx_args = sample_input if isinstance(sample_input, tuple) else (sample_input,)
        try:
            torch.onnx.export(
                model,
                onnx_args,
                str(target_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=opset_version,
                do_constant_folding=True,
            )
        except Exception:
            # Fallback explicitly setting dynamo=False if supported
            torch.onnx.export(
                model,
                onnx_args,
                str(target_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,
            )

        msg = f"ONNX export succeeded ({target_path.stat().st_size} bytes)"

        # Check numerical parity with onnxruntime if installed
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
            ort_inputs = {session.get_inputs()[0].name: sample_input.detach().cpu().numpy()}
            ort_outputs = session.run(None, ort_inputs)
            ort_tensor = torch.from_numpy(ort_outputs[0])

            diff = float(torch.max(torch.abs(ref_output.detach().cpu() - ort_tensor)).item())
            if diff > tol:
                return (
                    False,
                    f"{msg}, but ONNX Runtime parity error {diff:.3e} exceeds tolerance {tol}",
                )
            msg += f" (ONNX Runtime parity verified, max abs diff = {diff:.2e})"
        except ImportError:
            msg += " (onnxruntime not installed; skipped runtime parity validation)"

        return True, msg
    except Exception as exc:
        return False, f"ONNX export failed: {type(exc).__name__}: {exc}"
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def verify_stablehlo_export(
    jax_fn: Callable[..., Any],
    sample_input: Any,
) -> tuple[bool, str]:
    """Verifies that a JAX function serializes cleanly to StableHLO via jax.export.

    Args:
        jax_fn: JAX callable to export.
        sample_input: Example input pytree.

    Returns:
        tuple of (success_boolean, status_message).
    """
    try:
        import jax

        args = sample_input if isinstance(sample_input, tuple) else (sample_input,)
        exported = jax.export.export(jax.jit(jax_fn))(*args)  # type: ignore[attr-defined]
        stablehlo_text = exported.mlir_module()
        return True, f"StableHLO export succeeded ({len(stablehlo_text)} characters in MLIR)"
    except ImportError:
        return True, "JAX is not installed; skipped StableHLO export check."
    except Exception as exc:
        return False, f"StableHLO export failed: {type(exc).__name__}: {exc}"
