#!/usr/bin/env python3
"""
Self-Supervised Contrastive AST Expression Embedder Trainer.
Trains compact 8-dimensional semantic embeddings for mathematical AST nodes
(algebra-engine) using InfoNCE contrastive representation learning.
Exports the trained model to SafeTensors matching Candle VarBuilder schemas.
"""

from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file

from spodeian_ml.ssl.contrastive import InfoNCELoss


class ExpressionEmbedderModel(nn.Module):
    """
    Symbolic AST Node Embedder matching algebra-engine's NeuralExpressionEmbedder schema:
    - embeddings: [vocab_size=16, embed_dim=8]
    - projection: [embed_dim=8, embed_dim=8]
    """

    def __init__(self, vocab_size: int = 16, embed_dim: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # Initialize with structured orthogonal weights
        nn.init.orthogonal_(self.projection.weight)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch_size, seq_len]
        emb = self.embeddings(token_ids)  # [batch_size, seq_len, embed_dim]
        # Mean pool across tokens with linear projection
        pooled = torch.mean(emb, dim=1)  # [batch_size, embed_dim]
        proj = self.projection(pooled)
        # Normalize to unit sphere for cosine similarity proof guidance
        return nn.functional.normalize(proj, p=2, dim=-1)


def generate_synthetic_ast_pairs(batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthesize pairs of algebraically equivalent or structurally related AST expressions:
    Token IDs:
    0: PAD, 1: Number, 2: Symbol, 3: Add, 4: Mul, 5: Pow, 6: Div, 7: Sub, 8: Neg, 9: Function
    """
    # Equivalent patterns (commutativity, associativity, distribution)
    # View 1 and View 2 share algebraic semantics
    view1_list = []
    view2_list = []

    for _ in range(batch_size):
        pattern_type = torch.randint(0, 5, (1,)).item()
        if pattern_type == 0:
            # Add commutativity: x + y <=> y + x
            v1 = [3, 2, 2, 0]
            v2 = [3, 2, 2, 0]
        elif pattern_type == 1:
            # Mul commutativity: x * y <=> y * x
            v1 = [4, 2, 2, 0]
            v2 = [4, 2, 2, 0]
        elif pattern_type == 2:
            # Add identity: x + 0 <=> x
            v1 = [3, 2, 1, 0]
            v2 = [2, 0, 0, 0]
        elif pattern_type == 3:
            # Distributive: a * (b + c) <=> a*b + a*c
            v1 = [4, 2, 3, 2]
            v2 = [3, 4, 4, 2]
        else:
            # Power rule: x^2
            v1 = [5, 2, 1, 0]
            v2 = [4, 2, 2, 0]

        view1_list.append(v1)
        view2_list.append(v2)

    return torch.tensor(view1_list, dtype=torch.long), torch.tensor(view2_list, dtype=torch.long)


def train_expression_embedder(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_dir: Path = Path("models/algebra"),
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_safetensors = output_dir / "algebra_expression_embedder.safetensors"

    model = ExpressionEmbedderModel(vocab_size=16, embed_dim=8)
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        v1, v2 = generate_synthetic_ast_pairs(batch_size=batch_size)
        z1 = model(v1)
        z2 = model(v2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save to SafeTensors
    tensors = {
        "embeddings": model.embeddings.weight.detach().cpu().contiguous(),
        "projection": model.projection.weight.detach().cpu().contiguous(),
    }
    save_file(tensors, str(out_safetensors))

    manifest = {
        "model_type": "algebra_ast_contrastive_embedder",
        "vocab_size": 16,
        "embed_dim": 8,
        "final_loss": round(loss.item(), 4),
        "tensors": {
            "embeddings": list(tensors["embeddings"].shape),
            "projection": list(tensors["projection"].shape),
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_safetensors


if __name__ == "__main__":
    path = train_expression_embedder(epochs=20)
    print(f"Successfully trained & exported AST expression embedder to {path}")
