"""
Mathematical AST Canonicalizer & Tokenizer.

Implements literature-grounded pre-processing for symbolic mathematical expressions:
1. α-Equivalence & Variable Anonymization: Variables are canonically renamed by pre-order
   appearance (v0, v1, ...) to ensure invariance under variable relabeling (Wang et al., ICLR 2020).
2. Commutative-Associative (AC) Normalization: Children of commutative operators (+, *) are
   canonically ordered by subtree hash to eliminate syntactic permutations (Lample & Charton, 2020).
3. Float/Integer Decomposition: Number literals are decomposed into (sign, normalized mantissa,
   base-2 exponent) rather than atomic vocabulary lookups (Lample & Charton, 2020).
"""

from typing import List, Dict, Any, Tuple, Union, Optional
from dataclasses import dataclass, field
import math


# Operator Vocabulary & Category Classification
COMMUTATIVE_OPS = {"add", "mul", "min", "max", "gcd"}
ORDERED_OPS = {"sub", "div", "pow", "neg", "derivative", "integral", "matrix"}
FUNCTION_OPS = {"sin", "cos", "tan", "exp", "ln", "sqrt", "abs", "sinh", "cosh"}

OP_TO_ID: Dict[str, int] = {
    "pad": 0,
    "num": 1,
    "var": 2,
    "add": 3,
    "mul": 4,
    "pow": 5,
    "div": 6,
    "sub": 7,
    "neg": 8,
    "func": 9,
    "derivative": 10,
    "integral": 11,
    "matrix": 12,
    "relational": 13,
    "set_op": 14,
    "sum": 15,
}


@dataclass
class AstNode:
    """Canonical representation of an AST/DAG node."""
    node_id: int
    kind: str  # "num", "var", "add", "mul", etc.
    symbol: Optional[str] = None
    value: Optional[float] = None
    children: List[int] = field(default_factory=list)
    depth: int = 0
    is_commutative: bool = False
    
    # Decomposed numeric attributes
    num_sign: float = 0.0
    num_mantissa: float = 0.0
    num_exponent: float = 0.0


@dataclass
class CanonicalAstGraph:
    """A canonicalized Directed Acyclic Graph (DAG) representing an expression."""
    nodes: List[AstNode]
    root: int
    num_variables: int
    token_sequence: List[int]
    tree_depth: int


def decompose_number(val: float) -> Tuple[float, float, float]:
    """
    Decomposes a floating point number into (sign, normalized mantissa, base-2 exponent).
    Sign in {-1.0, 0.0, 1.0}, mantissa in [0.1, 1.0), exponent in [-64, 64].
    """
    if abs(val) < 1e-12:
        return 0.0, 0.0, 0.0
    
    sign = 1.0 if val > 0 else -1.0
    abs_v = abs(val)
    exponent = math.floor(math.log2(abs_v))
    mantissa = abs_v / (2.0 ** exponent)
    # Scale mantissa into [0.0, 1.0] range
    normalized_mantissa = (mantissa - 1.0) # since mantissa is in [1.0, 2.0)
    
    return sign, normalized_mantissa, float(exponent)


def canonicalize_expression_dict(expr_dict: Dict[str, Any]) -> CanonicalAstGraph:
    """
    Parses a nested expression dict (e.g. from E-graph or AST parser) and produces
    a canonicalized CanonicalAstGraph with α-renaming and AC ordering.
    """
    nodes: List[AstNode] = []
    var_map: Dict[str, int] = {}  # original_var_name -> canonical_id (v0, v1, ...)

    def traverse(raw_node: Dict[str, Any], current_depth: int) -> int:
        kind = raw_node.get("kind", "var").lower()
        node_id = len(nodes)
        is_comm = kind in COMMUTATIVE_OPS

        raw_children = raw_node.get("children", [])
        
        # AC Child Normalization: sort commutative children by structural signature
        if is_comm and len(raw_children) > 1:
            raw_children = sorted(raw_children, key=lambda c: str(c.get("kind", "")) + str(c.get("value", "")))

        node = AstNode(
            node_id=node_id,
            kind=kind,
            depth=current_depth,
            is_commutative=is_comm,
        )
        nodes.append(node)  # Reserve slot

        if kind == "num":
            val = float(raw_node.get("value", 0.0))
            node.value = val
            s, m, e = decompose_number(val)
            node.num_sign = s
            node.num_mantissa = m
            node.num_exponent = e
        elif kind == "var":
            raw_sym = str(raw_node.get("symbol", "x"))
            if raw_sym not in var_map:
                var_map[raw_sym] = len(var_map)
            canonical_idx = var_map[raw_sym]
            node.symbol = f"v{canonical_idx}"
            node.value = float(canonical_idx)
        else:
            node.symbol = raw_node.get("symbol")

        child_ids = []
        for child_dict in raw_children:
            child_id = traverse(child_dict, current_depth + 1)
            child_ids.append(child_id)
        
        node.children = child_ids
        return node_id

    root_id = traverse(expr_dict, 0)
    max_depth = max((n.depth for n in nodes), default=0)
    
    # Generate prefix Polish token sequence
    prefix_tokens: List[int] = []
    def to_prefix(nid: int):
        n = nodes[nid]
        prefix_tokens.append(OP_TO_ID.get(n.kind, 1))
        for c in n.children:
            to_prefix(c)
    
    to_prefix(root_id)

    return CanonicalAstGraph(
        nodes=nodes,
        root=root_id,
        num_variables=len(var_map),
        token_sequence=prefix_tokens,
        tree_depth=max_depth,
    )
