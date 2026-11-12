"""
Compiler Optimization & Graph Capture Engine for spodeian-ml.
Deploys PyTorch 2.x torch.compile with Inductor/Triton backend to capture
execution graphs, fuse memory-bound operator blocks (Layernorm + Activation + Linear),
and eliminate GPU memory bandwidth bottlenecks.
"""

from typing import Optional, Tuple, Any
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compile_model_for_inference(
    model: nn.Module,
    mode: str = "default",
    dynamic: bool = True,
    sample_input: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, dict[str, Any]]:
    """
    Optimizes a model using torch.compile with graceful fallback for environments
    lacking Triton/CUDA compilation backends.
    """
    meta: dict[str, Any] = {
        "compiled": False,
        "mode": mode,
        "dynamic": dynamic,
        "backend": "eager",
    }

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is not available in this PyTorch version. Returning eager model.")
        return model, meta

    try:
        compiled_model = torch.compile(model, mode=mode, dynamic=dynamic)
        meta["compiled"] = True
        meta["backend"] = "inductor"

        # Warmup run to trace CUDA graph if sample input provided
        if sample_input is not None:
            with torch.no_grad():
                _ = compiled_model(sample_input)
            meta["warmed_up"] = True
        else:
            meta["warmed_up"] = False

        return compiled_model, meta
    except Exception as e:
        logger.warning("torch.compile encountered error (%s). Falling back to eager model.", e)
        meta["fallback_reason"] = str(e)
        return model, meta
