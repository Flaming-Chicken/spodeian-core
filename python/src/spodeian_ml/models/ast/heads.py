"""
Multi-Task Prediction Heads for Symbolic AST Representations.

1. MctsValueHead: Estimates the normalized canonical distance / reward V_hat(E) in [0, 1].
2. RewritePolicyHead: Predicts the prior policy distribution pi(a | E) over candidate E-graph rewrite rules.
3. StructuralComplexityHead: Predicts target node count / tree depth after canonical reduction.
"""

import torch
import torch.nn as nn


class MctsValueHead(nn.Module):
    """Value Head estimating algebraic simplification progress V_hat(E) in [0, 1]."""
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, root_repr: torch.Tensor) -> torch.Tensor:
        return self.net(root_repr)


class RewritePolicyHead(nn.Module):
    """Policy Head outputting logits over discrete equality saturation rewrite rules."""
    def __init__(self, embed_dim: int, num_rules: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_rules),
        )

    def forward(self, root_repr: torch.Tensor) -> torch.Tensor:
        return self.net(root_repr)


class StructuralComplexityHead(nn.Module):
    """Estimates expected simplified node count C_hat(E) >= 1."""
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, root_repr: torch.Tensor) -> torch.Tensor:
        return self.net(root_repr)
