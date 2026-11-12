"""
Tree-Relative Transformer for Mathematical ASTs (MathTreeTransformer).

Implements the Shiv & Quirk (EMNLP 2019) and Wang et al. (2021) tree positional encoding:
- Computes tree-distance matrix D[i, j] (shortest path length on the AST).
- Computes ancestor-descendant relation matrix A[i, j] in {-1, 0, 1}.
- Injects tree-structural biases directly into the multi-head self-attention logits.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from spodeian_ml.models.ast.canonicalizer import CanonicalAstGraph, OP_TO_ID


def compute_tree_distance_matrix(graph: CanonicalAstGraph) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes (distance_matrix, ancestor_matrix) across all AST nodes.
    distance_matrix: (N, N) shortest undirected path length.
    ancestor_matrix: (N, N) where +1 means i is ancestor of j, -1 means i is descendant of j, 0 otherwise.
    """
    n = len(graph.nodes)
    dist = torch.full((n, n), fill_value=999, dtype=torch.long)
    ancestor = torch.zeros((n, n), dtype=torch.long)

    for i in range(n):
        dist[i, i] = 0

    # Build adjacency
    for node in graph.nodes:
        for cid in node.children:
            dist[node.node_id, cid] = 1
            dist[cid, node.node_id] = 1
            ancestor[node.node_id, cid] = 1
            ancestor[cid, node.node_id] = -1

    # Floyd-Warshall for shortest tree distance
    for k in range(n):
        dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

    # Transitive ancestor propagation
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if ancestor[i, k] == 1 and ancestor[k, j] == 1:
                    ancestor[i, j] = 1
                    ancestor[j, i] = -1

    return dist, ancestor


class TreeRelativeSelfAttention(nn.Module):
    """Multi-Head Attention with Tree-Relative Distance and Ancestor Biases."""
    def __init__(self, embed_dim: int, num_heads: int, max_tree_dist: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable structural bias tables (Shiv & Quirk 2019)
        self.dist_bias = nn.Embedding(max_tree_dist + 1, num_heads)
        self.ancestor_bias = nn.Embedding(3, num_heads)  # -1 -> 0, 0 -> 1, +1 -> 2

    def forward(
        self,
        x: torch.Tensor,
        tree_dist: torch.Tensor,
        ancestor_rel: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (1, N, embed_dim)
        tree_dist: (N, N)
        ancestor_rel: (N, N)
        """
        batch_size, n, _ = x.shape
        q = self.q_proj(x).view(batch_size, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (1, num_heads, N, N)

        # Add tree distance and ancestor biases
        clamped_dist = torch.clamp(tree_dist, 0, 16)
        dist_b = self.dist_bias(clamped_dist).permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)

        ancestor_idx = ancestor_rel + 1  # Map {-1, 0, 1} -> {0, 1, 2}
        ancestor_b = self.ancestor_bias(ancestor_idx).permute(2, 0, 1).unsqueeze(0)

        total_scores = scores + dist_b + ancestor_b
        attn_weights = F.softmax(total_scores, dim=-1)

        out = torch.matmul(attn_weights, v)  # (1, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, n, self.embed_dim)
        return self.out_proj(out)


class MathTreeTransformer(nn.Module):
    """Complete Tree-Relative Transformer for Mathematical ASTs."""
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        vocab_size: int = 32,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": TreeRelativeSelfAttention(embed_dim, num_heads),
                "ln1": nn.LayerNorm(embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dim),
                ),
                "ln2": nn.LayerNorm(embed_dim),
            })
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, graph: CanonicalAstGraph) -> torch.Tensor:
        device = self.token_embedding.weight.device
        dist_mat, anc_mat = compute_tree_distance_matrix(graph)
        dist_mat = dist_mat.to(device)
        anc_mat = anc_mat.to(device)

        tokens = torch.tensor([OP_TO_ID.get(n.kind, 1) for n in graph.nodes], device=device).unsqueeze(0)
        h = self.token_embedding(tokens)

        for layer in self.layers:
            attn_out = layer["attn"](h, dist_mat, anc_mat)
            h = layer["ln1"](h + attn_out)
            ffn_out = layer["ffn"](h)
            h = layer["ln2"](h + ffn_out)

        h = self.norm(h)
        # Root representation is at index graph.root
        root_repr = h[:, graph.root, :]
        return root_repr
