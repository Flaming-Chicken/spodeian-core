"""
Speculative Decoding Engine for spodeian-ml.
Deploys an ultra-fast, quantized draft model to autoregressively predict lookahead trajectories,
which are verified in parallel by the deep target alpha model in a single batched pass.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


class SpeculativeDecoder:
    """
    Coordinates speculative trajectory generation between a fast draft model
    and a deep target verification model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        gamma_lookahead: int = 4,
        tolerance: float = 0.05,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma_lookahead
        self.tolerance = tolerance

        self.total_drafted = 0
        self.total_accepted = 0

    def generate_step(
        self,
        prefix: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes speculative decoding:
        1. Draft model generates gamma lookahead tokens autoregressively.
        2. Target model evaluates the drafted tokens in a single parallel forward pass.
        3. Accepts drafted tokens within tolerance; resyncs upon divergence.
        """
        self.draft_model.eval()
        self.target_model.eval()

        device = prefix.device
        current_seq = prefix.clone()

        seq_dim = 1 if prefix.dim() == 3 else 0

        with torch.no_grad():
            # 1. Draft phase
            draft_tokens = []
            draft_in = current_seq.clone()
            for _ in range(self.gamma):
                out = self.draft_model(draft_in)
                if prefix.dim() == 3:
                    next_val = out[:, -1:, :]
                else:
                    next_val = out[-1:, :] if out.dim() == 2 else out.unsqueeze(0)
                draft_tokens.append(next_val)
                draft_in = torch.cat([draft_in, next_val], dim=seq_dim)

            # 2. Parallel verification pass on target model
            target_out = self.target_model(draft_in)
            if prefix.dim() == 3:
                target_preds = target_out[:, -self.gamma:, :]
            else:
                target_preds = target_out[-self.gamma:, :]

            # 3. Acceptance evaluation
            accepted_tokens = []
            accepted_count = 0

            for i in range(len(draft_tokens)):
                d_token = draft_tokens[i]
                if prefix.dim() == 3:
                    t_token = target_preds[:, i:i+1, :]
                else:
                    t_token = target_preds[i:i+1, :]

                err = torch.norm(d_token - t_token).item()

                if err <= self.tolerance:
                    accepted_tokens.append(d_token)
                    accepted_count += 1
                else:
                    # Reject remaining drafted tokens upon divergence; accept target's correction
                    accepted_tokens.append(t_token)
                    break

            self.total_drafted += self.gamma
            self.total_accepted += accepted_count

            acc_rate = (self.total_accepted / max(1, self.total_drafted)) * 100.0
            meta = {
                "gamma": self.gamma,
                "accepted_steps": accepted_count,
                "cumulative_acceptance_rate_pct": round(acc_rate, 2),
            }

            result = torch.cat([prefix] + accepted_tokens, dim=seq_dim)
            return result, meta
