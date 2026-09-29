"""
Generalization Smoothing & Weight Averaging for spodeian-ml.
Implements Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA)
to settle weights into flat loss basins, improving out-of-distribution generalization.
"""

from typing import Dict
import torch
import torch.nn as nn


class ModelWeightAverager:
    """
    Maintains a shadow copy of model weights using Exponential Moving Average (EMA)
    or Stochastic Weight Averaging (SWA).
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "ema",
        decay: float = 0.999,
    ):
        self.model = model
        self.mode = mode.lower()
        self.decay = decay
        self.num_averaged = 0

        # Shadow parameter dictionary
        self.shadow_params: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow_params[name] = p.detach().clone()

    def update(self) -> None:
        """Updates shadow parameters based on current model weights."""
        self.num_averaged += 1
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.shadow_params:
                    if self.mode == "swa":
                        # SWA uniform average: w_avg = (w_avg * (n-1) + w) / n
                        n = self.num_averaged
                        self.shadow_params[name].mul_((n - 1) / n).add_(p.detach() / n)
                    else:
                        # EMA geometric decay: w_avg = decay * w_avg + (1 - decay) * w
                        d = self.decay
                        self.shadow_params[name].mul_(d).add_(p.detach() * (1.0 - d))

    def apply_shadow_weights(self) -> None:
        """Copies shadow weights into the active model parameters."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.shadow_params:
                    p.copy_(self.shadow_params[name])
