"""
Contrastive Loss Primitives for Self-Supervised Learning.
Implements InfoNCE / NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss between two paired augmented representations (view_a, view_b).
    Pulls positive pairs together in latent space while repelling negative pairs across the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        z_a: (batch_size, latent_dim)
        z_b: (batch_size, latent_dim)
        """
        batch_size = z_a.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)

        # L2 normalize embeddings
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        # Concatenate both views: (2 * batch_size, latent_dim)
        representations = torch.cat([z_a, z_b], dim=0)

        # Compute cosine similarity matrix: (2B, 2B)
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Create labels: each positive pair is shifted by batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_a.device),
            torch.arange(0, batch_size, device=z_a.device),
        ], dim=0)

        # Mask out self-similarity (diagonal)
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_a.device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, float("-inf"))

        loss = F.cross_entropy(sim_matrix, labels)
        return loss
