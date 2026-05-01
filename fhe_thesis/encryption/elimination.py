"""Word/token elimination for FHE Transformer inference.

Background
----------
BOLT (S&P 2024) introduced "Word Elimination" (W.E.) — a 2x speedup
trick where the bottom 50% of tokens by Q@K^T attention score are
dropped after the first attention layer, and the remaining tokens
proceed through a half-size network.  BOLT relies on MPC + oblivious
bitonic sort to keep elimination *content-private*; that primitive is
prohibitively expensive in pure FHE (sorting under CKKS would blow the
multiplicative-depth budget many times over).

Adaptation for HyPER-LPAN (pure FHE, semi-honest server)
--------------------------------------------------------
We provide two strategies, ordered from "free + safe" to
"powerful but with caveats":

1. ``padding_elimination`` — drop ``[PAD]`` tokens **before
   encryption** based on the client-side attention mask.  This is
   lossless (FHE attention has no mask, so padding tokens otherwise
   contaminate Q@K^T), client-side (server learns nothing it didn't
   already know — the count of ciphertexts already reveals padded
   length in BOLT and in our setting), and gives the bulk of the
   end-to-end win on short tasks (RTE: 64→avg 30, MRPC: 128→avg 50).

2. ``content_elimination_teacher`` — drop low-attention tokens chosen
   *by a plaintext teacher model on the client*.  The client computes
   the score from its own copy of the input, sends the kept-indices
   alongside the ciphertexts.  This is content-aware but reveals the
   keep mask to the server.  Use only with semi-honest threat model
   and when the caller explicitly opts in.

Public API
----------
keep_indices_from_mask(attention_mask)
    Trivially convert HF attention_mask -> indices array of non-pad
    positions.

apply_elimination(embeddings, attention_mask, *, strategy=...,
                  teacher_scores=None, keep_ratio=0.5)
    Return (filtered_embeddings, kept_indices) ready for encryption.

slice_pos_mix(P_weights, kept_indices)
    Sub-matrix select for LinearMixing layers when non-prefix indices
    are kept.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

EliminationStrategy = Literal["padding", "content_teacher", "none"]


def keep_indices_from_mask(attention_mask: np.ndarray) -> np.ndarray:
    """Return indices where the HuggingFace attention_mask is non-zero.

    Parameters
    ----------
    attention_mask : np.ndarray
        Shape ``(seq_len,)`` or ``(1, seq_len)``; 1 = real token, 0 = pad.

    Returns
    -------
    np.ndarray
        Sorted 1-D int array of kept positions.
    """
    m = np.asarray(attention_mask).reshape(-1)
    return np.flatnonzero(m).astype(np.int64)


def keep_indices_from_scores(
    scores: np.ndarray,
    keep_ratio: float = 0.5,
    *,
    always_keep_cls: bool = True,
) -> np.ndarray:
    """Return indices of top ``keep_ratio`` tokens by score, sorted ascending.

    Parameters
    ----------
    scores : np.ndarray
        Per-token contribution scores (higher = keep), shape ``(seq_len,)``.
    keep_ratio : float
        Fraction of tokens to keep.  ``0.5`` matches BOLT's median rule.
    always_keep_cls : bool
        Force-keep position 0 even if it scores low.  Required for
        sequence-classification heads that pool from CLS.

    Returns
    -------
    np.ndarray
        Sorted ascending int array of kept positions, preserving original
        relative order (matching BOLT's index-tag trick in Alg. 2).
    """
    s = np.asarray(scores).reshape(-1)
    n = len(s)
    k = max(1, int(round(n * keep_ratio)))
    # argpartition is O(n); the top-k indices are unsorted, so we sort
    # them ascending to preserve original token order downstream.
    top = np.argpartition(-s, k - 1)[:k]
    if always_keep_cls and 0 not in top:
        # Replace the lowest-scoring kept token with position 0
        worst_in_top = top[np.argmin(s[top])]
        top = top[top != worst_in_top]
        top = np.concatenate([top, [0]])
    return np.sort(top).astype(np.int64)


def apply_elimination(
    embeddings: np.ndarray,
    attention_mask: Optional[np.ndarray] = None,
    *,
    strategy: EliminationStrategy = "padding",
    teacher_scores: Optional[np.ndarray] = None,
    keep_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter the embedding tensor according to the chosen strategy.

    Parameters
    ----------
    embeddings : np.ndarray
        Plaintext embedding output, shape ``(seq_len, hidden_dim)``.
    attention_mask : np.ndarray, optional
        Required for ``strategy="padding"``.
    strategy : {"padding", "content_teacher", "none"}
        ``padding`` (default, free + safe): drop tokens where
        ``attention_mask == 0``.
        ``content_teacher``: drop bottom ``1 - keep_ratio`` tokens by
        ``teacher_scores`` (client-side).
        ``none``: identity, returns inputs unchanged with full index set.
    teacher_scores : np.ndarray, optional
        Required for ``strategy="content_teacher"``; per-token scores.
    keep_ratio : float
        Used only for ``content_teacher``.

    Returns
    -------
    (filtered_embeddings, kept_indices)
        ``filtered_embeddings`` shape ``(k, hidden_dim)`` where ``k`` is
        the number of kept tokens.  ``kept_indices`` is the int array
        used to slice ``LinearMixing.P_weights`` later.
    """
    if strategy == "none":
        n = embeddings.shape[0]
        return embeddings, np.arange(n, dtype=np.int64)

    if strategy == "padding":
        if attention_mask is None:
            raise ValueError("padding elimination requires attention_mask")
        keep = keep_indices_from_mask(attention_mask)
    elif strategy == "content_teacher":
        if teacher_scores is None:
            raise ValueError("content_teacher elimination requires teacher_scores")
        # First clip to padding region (don't keep [PAD] even if score is high)
        if attention_mask is not None:
            valid = keep_indices_from_mask(attention_mask)
            scores_valid = np.zeros_like(teacher_scores, dtype=np.float64)
            scores_valid[valid] = teacher_scores[valid]
            keep = keep_indices_from_scores(scores_valid, keep_ratio=keep_ratio)
            # Intersect with valid (in case keep_ratio > valid_fraction)
            keep = np.array(sorted(set(keep.tolist()) & set(valid.tolist())), dtype=np.int64)
        else:
            keep = keep_indices_from_scores(teacher_scores, keep_ratio=keep_ratio)
    else:
        raise ValueError(f"Unknown elimination strategy: {strategy!r}")

    return embeddings[keep], keep


def slice_pos_mix(P_weights: np.ndarray, kept_indices: np.ndarray) -> np.ndarray:
    """Sub-matrix select position-mixing weights along token axes.

    LinearMixing layers carry a per-head ``(H, max_seq_len, max_seq_len)``
    position-mixing tensor.  When word-elimination keeps a non-prefix
    set of token positions, both the row axis (output token) and the
    column axis (input token) need to be sub-selected to the kept
    indices, preserving their relative order.

    Parameters
    ----------
    P_weights : np.ndarray
        Shape ``(H, max_seq_len, max_seq_len)``.
    kept_indices : np.ndarray
        Sorted 1-D int array, length ``k``.

    Returns
    -------
    np.ndarray
        Shape ``(H, k, k)``.
    """
    return P_weights[:, kept_indices, :][:, :, kept_indices]


def slice_pos_bias(P_biases: np.ndarray, kept_indices: np.ndarray) -> np.ndarray:
    """Sub-vector select position-mixing biases.

    Parameters
    ----------
    P_biases : np.ndarray
        Shape ``(H, max_seq_len)``.
    kept_indices : np.ndarray
        Sorted 1-D int array, length ``k``.

    Returns
    -------
    np.ndarray
        Shape ``(H, k)``.
    """
    return P_biases[:, kept_indices]


def elimination_savings(orig_seq_len: int, kept: int) -> dict:
    """Report theoretical wall-clock savings from elimination.

    Costs that scale linearly with seq_len:
      - encrypt: O(L) ciphertexts produced
      - LinearMixing pt*ct loop: O(L^2) (each output token = sum over inputs)
      - QuadAttention QK^T:      O(L^2)
      - Attention apply A*V:     O(L^2)
      - All LN, residual, FFN:   O(L)

    The L^2 terms dominate, so a ratio L/L_orig delivers roughly
    (L/L_orig)^2 reduction on attention-block work, and L/L_orig on
    point-wise (FFN, LN) work.
    """
    if orig_seq_len <= 0:
        return {"keep_ratio": 0.0, "linear_speedup": 0.0, "quadratic_speedup": 0.0}
    r = kept / orig_seq_len
    return {
        "kept_tokens": int(kept),
        "orig_tokens": int(orig_seq_len),
        "keep_ratio": float(r),
        "linear_speedup": float(1.0 / r) if r > 0 else float("inf"),
        "quadratic_speedup": float(1.0 / (r * r)) if r > 0 else float("inf"),
    }
