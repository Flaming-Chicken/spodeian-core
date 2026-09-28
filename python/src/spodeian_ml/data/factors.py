"""
Quantitative factor indicators and feature extractors.
Powered by spodeian_py_bridge for 1:1 mathematical parity with the Rust production engine.
"""

from typing import Union, Sequence
import torch

try:
    from spodeian_ml import spodeian_py_bridge
except ImportError:
    import spodeian_py_bridge


def log_returns(prices: Union[torch.Tensor, Sequence[float]]) -> torch.Tensor:
    """Computes log returns: ln(P_t / P_{t-1}). First return is 0.0."""
    if isinstance(prices, torch.Tensor):
        device = prices.device
        dtype = prices.dtype
        vals = prices.detach().cpu().view(-1).tolist()
        res = spodeian_py_bridge.py_log_returns(vals)
        return torch.tensor(res, dtype=dtype, device=device).view_as(prices)
    else:
        vals = list(prices)
        res = spodeian_py_bridge.py_log_returns(vals)
        return torch.tensor(res, dtype=torch.float32)


def rolling_volatility(returns: Union[torch.Tensor, Sequence[float]], window: int = 20) -> torch.Tensor:
    """Computes sample rolling volatility annualized with sqrt(252)."""
    if isinstance(returns, torch.Tensor):
        device = returns.device
        dtype = returns.dtype
        vals = returns.detach().cpu().view(-1).tolist()
        res = spodeian_py_bridge.py_rolling_volatility(vals, int(window))
        return torch.tensor(res, dtype=dtype, device=device).view_as(returns)
    else:
        vals = list(returns)
        res = spodeian_py_bridge.py_rolling_volatility(vals, int(window))
        return torch.tensor(res, dtype=torch.float32)


def exponential_moving_average(series: Union[torch.Tensor, Sequence[float]], span: int = 20) -> torch.Tensor:
    """Computes Exponential Moving Average (EMA)."""
    if isinstance(series, torch.Tensor):
        device = series.device
        dtype = series.dtype
        vals = series.detach().cpu().view(-1).tolist()
        res = spodeian_py_bridge.py_ema(vals, int(span))
        return torch.tensor(res, dtype=dtype, device=device).view_as(series)
    else:
        vals = list(series)
        res = spodeian_py_bridge.py_ema(vals, int(span))
        return torch.tensor(res, dtype=torch.float32)


def vwap(
    prices: Union[torch.Tensor, Sequence[float]],
    volumes: Union[torch.Tensor, Sequence[float]],
) -> torch.Tensor:
    """Computes cumulative Volume-Weighted Average Price (VWAP)."""
    p_vals = prices.detach().cpu().view(-1).tolist() if isinstance(prices, torch.Tensor) else list(prices)
    v_vals = volumes.detach().cpu().view(-1).tolist() if isinstance(volumes, torch.Tensor) else list(volumes)
    res = spodeian_py_bridge.py_vwap(p_vals, v_vals)
    return torch.tensor(res, dtype=torch.float32)
