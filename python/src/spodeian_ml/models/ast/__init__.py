"""
Symbolic Mathematical AST & Graph Neural Representation Learning.

Grounding in relevant literature:
- Child-Sum & N-Ary Tree-LSTM Compositionality (Tai et al., ACL 2015)
- Tree-Relative Positional Encodings (Shiv & Quirk, EMNLP 2019; Wang et al., 2021)
- Deep Learning for Symbolic Mathematics (Lample & Charton, ICLR 2020)
- Directed Acyclic Graph Neural Networks (Thost & Chen, ICLR 2021)
- Neural Equality Saturation & E-Graph Value Guidance (Koehler et al., PLDI 2024 / NeurIPS 2023)
"""

from spodeian_ml.models.ast.canonicalizer import (
    CanonicalAstGraph,
    AstNode,
    canonicalize_expression_dict,
    decompose_number,
    OP_TO_ID,
    COMMUTATIVE_OPS,
    ORDERED_OPS,
)
from spodeian_ml.models.ast.dag_encoder import MathDAGEncoder
from spodeian_ml.models.ast.tree_encoder import MathTreeTransformer, compute_tree_distance_matrix
from spodeian_ml.models.ast.heads import MctsValueHead, RewritePolicyHead, StructuralComplexityHead
from spodeian_ml.models.ast.loss import EquivalenceContrastiveLoss, JointAstTrainingLoss

__all__ = [
    "CanonicalAstGraph",
    "AstNode",
    "canonicalize_expression_dict",
    "decompose_number",
    "OP_TO_ID",
    "COMMUTATIVE_OPS",
    "ORDERED_OPS",
    "MathDAGEncoder",
    "MathTreeTransformer",
    "compute_tree_distance_matrix",
    "MctsValueHead",
    "RewritePolicyHead",
    "StructuralComplexityHead",
    "EquivalenceContrastiveLoss",
    "JointAstTrainingLoss",
]
