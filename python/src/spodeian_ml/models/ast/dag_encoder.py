"""
Operator-Aware Directed Acyclic Graph Neural Network (MathDAGEncoder).

Implements literature-grounded bottom-up compositional message passing for mathematical expressions:
1. Commutative Child-Sum Pooling (Tai et al., ACL 2015):
   For AC operators (+, *), computes permutation-invariant sum-aggregation over children.
2. Positional Gated Projection (Tai et al., ACL 2015; Wang et al., ICLR 2020):
   For non-commutative operators (-, /, pow, derivative), applies order-specific linear transforms.
3. Topological Propagation (Thost & Chen, ICLR 2021):
   Executes in strict reverse topological order (leaves -> root) with zero recursion overhead.
"""

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from spodeian_ml.models.ast.canonicalizer import CanonicalAstGraph, OP_TO_ID, COMMUTATIVE_OPS, ORDERED_OPS


class MathDAGEncoder(nn.Module):
    """
    Operator-Aware Recursive DAG Neural Network for Mathematical ASTs.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 32,
        max_arity: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_arity = max_arity
        
        # 1. Base Node Type Embeddings
        self.op_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Continuous Number Feature Projection: [sign, mantissa, exponent] -> embed_dim
        self.num_projection = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # 3. Variable Canonical ID Embedding (v0, v1, ..., v15)
        self.var_embedding = nn.Embedding(32, embed_dim)
        
        # 4. Commutative Aggregator (Permutation-Invariant Child-Sum)
        self.comm_child_transform = nn.Linear(embed_dim, embed_dim)
        
        # 5. Non-Commutative Positional Argument Transforms (N-ary Tree Weights)
        self.pos_transforms = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(max_arity)
        ])
        
        # 6. Node Composition Cell
        self.cell = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: CanonicalAstGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes bottom-up topological message passing across the AST graph.
        Returns:
            root_embedding: (1, embed_dim) vector embedding of the complete expression
            all_node_embeddings: (num_nodes, embed_dim) tensor of all intermediate node states
        """
        device = self.op_embedding.weight.device
        num_nodes = len(graph.nodes)
        node_states = torch.zeros((num_nodes, self.embed_dim), device=device)

        # Bottom-up evaluation order: reverse sorted by depth
        evaluation_order = sorted(range(num_nodes), key=lambda i: graph.nodes[i].depth, reverse=True)

        for nid in evaluation_order:
            node = graph.nodes[nid]
            op_id = torch.tensor([OP_TO_ID.get(node.kind, 1)], device=device)
            base_repr = self.op_embedding(op_id).squeeze(0)  # (embed_dim,)

            # Incorporate literal / variable continuous features
            if node.kind == "num":
                num_features = torch.tensor([node.num_sign, node.num_mantissa, node.num_exponent], device=device).unsqueeze(0)
                num_repr = self.num_projection(num_features).squeeze(0)
                base_repr = base_repr + num_repr
            elif node.kind == "var":
                var_idx = min(int(node.value or 0), 31)
                var_repr = self.var_embedding(torch.tensor([var_idx], device=device)).squeeze(0)
                base_repr = base_repr + var_repr

            # Child aggregation
            if not node.children:
                # Leaf node
                node_states[nid] = self.layer_norm(base_repr)
            else:
                child_reprs = [node_states[cid] for cid in node.children]

                if node.is_commutative:
                    # Permutation-invariant sum aggregation
                    stacked = torch.stack(child_reprs, dim=0)
                    summed = torch.sum(self.comm_child_transform(stacked), dim=0)
                    child_ctx = summed
                else:
                    # Ordered positional projection (N-ary weights)
                    child_ctx = torch.zeros(self.embed_dim, device=device)
                    for pos_idx, c_state in enumerate(child_reprs[:self.max_arity]):
                        child_ctx = child_ctx + self.pos_transforms[pos_idx](c_state)

                # Fuse base node representation with child context
                fused = torch.cat([base_repr, child_ctx], dim=-1).unsqueeze(0)
                next_state = self.cell(fused).squeeze(0)
                node_states[nid] = self.layer_norm(base_repr + self.dropout(next_state))

        root_embedding = node_states[graph.root].unsqueeze(0)
        return root_embedding, node_states
