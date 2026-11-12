"""Pure-CPU vectorized hardware kernels for Category 01 Topic 07:
07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds
"""

from __future__ import annotations

import numpy as np


def rotation_6d_to_matrix_cpu(d6: np.ndarray) -> np.ndarray:
    """Vectorized CPU kernel mapping 6D representation to SO(3) 3x3 rotation matrix."""
    d6_arr = np.asarray(d6, dtype=np.float32)
    a1 = d6_arr[..., 0:3]
    a2 = d6_arr[..., 3:6]

    norm_a1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / np.maximum(norm_a1, 1e-8)

    dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
    u2 = a2 - dot_prod * b1
    norm_u2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    b2 = u2 / np.maximum(norm_u2, 1e-8)

    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def pointer_network_forward_cpu(
    encoder_seq: np.ndarray,
    decoder_state: np.ndarray,
    w_enc: np.ndarray,
    w_dec: np.ndarray,
    b_ptr: np.ndarray,
    v_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized CPU kernel computing pointer logits and softmax distribution."""
    enc = np.asarray(encoder_seq, dtype=np.float32)
    dec = np.asarray(decoder_state, dtype=np.float32)
    we = np.asarray(w_enc, dtype=np.float32)
    wd = np.asarray(w_dec, dtype=np.float32)
    bp = np.asarray(b_ptr, dtype=np.float32)
    vp = np.asarray(v_ptr, dtype=np.float32)

    e_proj = np.dot(enc, we)
    d_proj = np.expand_dims(np.dot(dec, wd), axis=1)
    u_scores = np.tanh(e_proj + d_proj + bp)
    logits = np.squeeze(np.dot(u_scores, vp), axis=-1)

    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return logits, probs


def fourier_coordinate_encode_cpu(coords: np.ndarray, num_bands: int = 4) -> np.ndarray:
    """Vectorized CPU kernel mapping continuous coords to multi-frequency Fourier embeddings."""
    c_arr = np.asarray(coords, dtype=np.float32)
    scales = (2.0 ** np.arange(num_bands, dtype=np.float32)) * np.pi
    scaled = np.expand_dims(c_arr, axis=-1) * np.reshape(scales, (1, 1, -1))
    sin_part = np.sin(scaled)
    cos_part = np.cos(scaled)
    out = np.stack([sin_part, cos_part], axis=-1)
    return np.reshape(out, (c_arr.shape[0], -1))


def patch_embedding_forward_cpu(patches: np.ndarray, w_patch: np.ndarray, b_patch: np.ndarray) -> np.ndarray:
    """Vectorized CPU kernel for ViT patch linear embedding."""
    p_arr = np.asarray(patches, dtype=np.float32)
    wp = np.asarray(w_patch, dtype=np.float32)
    bp = np.asarray(b_patch, dtype=np.float32)
    return np.dot(p_arr, wp) + bp
