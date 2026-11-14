"""Drop-in PyTorch Optimizer and SciPy solve_ivp wrappers backed by `spodeian_py_bridge`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

try:
    import spodeian_py_bridge as _bridge
except ImportError as exc:  # pragma: no cover - native extension built via maturin
    _bridge = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_bridge():
    if _bridge is None:
        raise ImportError(
            "spodeian_py_bridge native extension is not built; run `maturin develop` in crates/sml-py-bridge"
        ) from _IMPORT_ERROR
    return _bridge


class SpodeianMetaOptimizer:
    """Drop-in `torch.optim.Optimizer`-compatible wrapper around `PySelfOptimizingMeta`."""

    def __init__(self, params: Iterable, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        bridge = _require_bridge()
        self.param_groups = [{"params": list(params), "lr": lr, "weight_decay": weight_decay}]
        self._opts = [bridge.PySelfOptimizingMeta(lr=lr, weight_decay=weight_decay) for _ in self.param_groups[0]["params"]]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure: Callable[[], float] | None = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p, opt in zip(group["params"], self._opts):
                if p.grad is None:
                    continue
                flat_p = p.data.detach().reshape(-1).cpu().tolist()
                flat_g = p.grad.detach().reshape(-1).cpu().tolist()
                updated = opt.step(flat_p, flat_g)
                p.data.copy_(p.data.new_tensor(updated).reshape_as(p.data))
        return loss


@dataclass
class MetaSolveIvpResult:
    t: Tuple[float, float]
    y_final: list[float]
    nfev: int
    success: bool = True


def solve_ivp_meta(
    fun: Callable[[float, Sequence[float]], Sequence[float]],
    t_span: Tuple[float, float],
    y0: Sequence[float],
    rtol: float = 1e-4,
) -> MetaSolveIvpResult:
    """`scipy.integrate.solve_ivp`-compatible wrapper around `SelfOptimizingMetaIntegrator`."""
    bridge = _require_bridge()
    t0, t1 = float(t_span[0]), float(t_span[1])
    xf, nfev = bridge.py_meta_integrate(
        [float(v) for v in y0],
        t0,
        t1,
        lambda state, t: [float(v) for v in fun(t, state)],
        tol=float(rtol),
    )
    return MetaSolveIvpResult(t=(t0, t1), y_final=list(xf), nfev=int(nfev), success=True)
