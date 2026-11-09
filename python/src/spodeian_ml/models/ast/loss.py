"""
Loss Functions for Provable Symbolic Representation Learning.

1. EquivalenceContrastiveLoss:
   InfoNCE contrastive loss over algebraically equivalent AST pairs (positive pairs)
   and structurally similar but semantically distinct expressions refuted by
   Schwartz-Zippel randomized polynomial testing (hard negative pairs).
2. JointAstTrainingLoss:
   Joint multi-task objective combining contrastive representation learning,
   MCTS value regression, and policy cross-entropy.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class EquivalenceContrastiveLoss(nn.Module):
    """InfoNCE loss aligning mathematically equivalent expression embeddings."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_embeddings: torch.Tensor,     # (B, D)
        equivalent_embeddings: torch.Tensor, # (B, D)
        negative_embeddings: Optional[torch.Tensor] = None, # (B, K, D)
    ) -> torch.Tensor:
        # Normalize to unit sphere
        z_a = F.normalize(anchor_embeddings, p=2, dim=-1)
        z_pos = F.normalize(equivalent_embeddings, p=2, dim=-1)

        # Positive cosine similarities
        pos_sim = torch.sum(z_a * z_pos, dim=-1, keepdim=True) / self.temperature  # (B, 1)

        if negative_embeddings is not None:
            z_neg = F.normalize(negative_embeddings, p=2, dim=-1)
            # Batch matrix multiplication: (B, 1, D) @ (B, D, K) -> (B, 1, K)
            neg_sim = torch.bmm(z_a.unsqueeze(1), z_neg.transpose(1, 2)).squeeze(1) / self.temperature  # (B, K)
            logits = torch.cat([pos_sim, neg_sim], dim=-1)  # (B, 1 + K)
            labels = torch.zeros(z_a.shape[0], dtype=torch.long, device=z_a.device)
            return F.cross_entropy(logits, labels)
        else:
            # Self-contrast across the batch (in-batch negatives)
            sim_matrix = torch.matmul(z_a, z_pos.t()) / self.temperature  # (B, B)
            labels = torch.arange(z_a.shape[0], device=z_a.device)
            return F.cross_entropy(sim_matrix, labels)


class JointAstTrainingLoss(nn.Module):
    """Composite objective for self-supervised equality saturation guidance."""
    def __init__(self, contrastive_weight: float = 1.0, value_weight: float = 0.5, policy_weight: float = 0.5):
        super().__init__()
        self.contrastive_loss = EquivalenceContrastiveLoss()
        self.value_loss_fn = nn.SmoothL1Loss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.contrastive_weight = contrastive_weight
        self.value_weight = value_weight
        self.policy_weight = policy_weight

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_equivalent: torch.Tensor,
        predicted_value: torch.Tensor,
        target_value: torch.Tensor,
        predicted_policy_logits: Optional[torch.Tensor] = None,
        target_rule_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_c = self.contrastive_loss(z_anchor, z_equivalent)
        loss_v = self.value_loss_fn(predicted_value, target_value)

        total_loss = self.contrastive_weight * loss_c + self.value_weight * loss_v

        if predicted_policy_logits is not None and target_rule_action is not None:
            loss_p = self.policy_loss_fn(predicted_policy_logits, target_rule_action)
            total_loss = total_loss + self.policy_weight * loss_p

        return total_loss
