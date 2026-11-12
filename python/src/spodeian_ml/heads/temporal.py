"""
Temporal Sequential Head.
Connects latent sequence representations across time using bidirectional GRU or Transformer decoders.
"""

import torch
import torch.nn as nn


class TemporalHead(nn.Module):
    """Recurrent temporal sequential head modeling momentum and multi-horizon dynamics."""

    def __init__(self, in_features: int, hidden_dim: int = 64, num_layers: int = 2, out_features: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 2D, unsqueeze sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        # Take representation at final time step
        last_step = out[:, -1, :]
        return self.fc(last_step)
