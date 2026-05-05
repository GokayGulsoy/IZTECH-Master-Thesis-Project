"""Synthesizer-LPAN attention (Tay 2020) - first FHE port. O = A * V where A is plaintext.

Sliced from the original ``ops_attention_nexus.py`` during the production
re-modularization (synthesizer-lpan-production branch).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext

# -------------------------------------------------------------------------
# Synthesizer-LPAN attention (Tay 2020) - first FHE port. O = A * V where A is plaintext.
# -------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Synthesizer-LPAN: dense fixed-pattern attention (NO Q, NO K, NO ct·ct)
# ---------------------------------------------------------------------------
#
# Mathematical contract
# ---------------------
# Tay et al. (NeurIPS 2020) "Synthesizer: Rethinking Self-Attention"
# showed that a *learned, frozen* attention pattern A ∈ R^{L×L}
# (with softmax already absorbed at training time) recovers > 97% of
# standard-MHA accuracy on GLUE while completely eliminating the
# dynamic dependence on Q,K. Per-head:
#     O = A · V,      A ∈ R^{L×L}  is plaintext, learned, layer-specific.
#
# Under FHE this is transformative:
#   • Wq, Wk linears: ELIMINATED (no Q, no K).
#   • qk_scores:     ELIMINATED (no Q·K^T).
#   • softmax-poly:  ELIMINATED (already absorbed into A at train time).
#   • av:            REPLACED by pt·ct primitive — every ct·ct mul on
#                    the attention critical path becomes a mul_plain.
#
# This is the architectural lever the LPAN-family papers did not take
# because plaintext Synthesizer offers little speedup. Under FHE the
# entire L²-ct·ct floor disappears.
#
# Layout (this implementation)
# ----------------------------
# V_ct: NEXUS column-major, slot[h·H + j·L + i] = V[i, j], H = head_dim·L.
# A_per_head: np.ndarray of shape (num_heads_per_ct, L, L). A[h, i, t] is
#   the (already softmax-absorbed) attention weight from query position i
#   to key position t in head h.
# Output: same NEXUS col-major shape as V_ct,
#   slot[h·H + j·L + i] = Σ_t A[h, i, t] · V[t, j].
#
# Algorithm — diagonal decomposition (mirrors attn_apply_nexus structure):
#   For d ∈ [0, L):
#     diag_mask_d:  slot[h·H + j·L + i] = A[h, i, (i+d) mod L]
#                    (broadcast across all j ∈ [0, head_dim) within head h)
#     V_shift_d  :  cyclic shift of V along i-axis by d (existing trick)
#     O += mul_plain(V_shift_d, diag_mask_d)
#
# Cost per call (per V_ct head-bundle):
#   - V cyclic shift: same as baseline attn_apply_nexus (L iterations of
#     2 rot + 2 mul_plain + 1 add)  ≈ 5 × L ops.
#   - Diagonal apply:  L iterations of (1 mul_plain + 1 add)  = 2 × L ops.
#   - Output broadcast loop ELIMINATED (saves L · log2(head_dim) rotations
#     vs baseline = ~768 ops for L=128, head_dim=64).
#   - Zero ct·ct multiplications (vs L = 128 in baseline).
#
# Cache: A_per_head is data-independent ⇒ all L diagonals are encoded
# ONCE at model-load time and reused across every inference. Per-inference
# encoding cost is zero.
# ---------------------------------------------------------------------------


def encode_synthesizer_diagonals(
    backend: CKKSBackend,
    A_per_head: np.ndarray,
    *,
    L: int,
    head_dim: int,
    num_heads_per_ct: int = 1,
) -> List:
    """Pre-encode the L plaintext diagonals of a Synthesizer attention pattern.

    Parameters
    ----------
    A_per_head : np.ndarray, shape (num_heads_per_ct, L, L)
        ``A[h, i, t]`` = (softmax-absorbed) attention weight from query
        token i to key token t in head h. Heads packed in this bundle
        share one ciphertext.

    Returns
    -------
    List[Plaintext] of length L. Entry d holds the plaintext mask
        slot[h·H + j·L + i] = A_per_head[h, i, (i+d) mod L]
    broadcast across j ∈ [0, head_dim) within each head region.
    """
    n_slots = backend.capabilities.n_slots
    H = head_dim * L
    total = num_heads_per_ct * H
    if total > n_slots:
        raise ValueError(
            f"num_heads_per_ct·L·head_dim={total} > num_slots={n_slots}"
        )
    if A_per_head.shape != (num_heads_per_ct, L, L):
        raise ValueError(
            f"A_per_head shape {A_per_head.shape} != "
            f"({num_heads_per_ct}, {L}, {L})"
        )

    diag_pts = []
    for d in range(L):
        mask = [0.0] * n_slots
        for h in range(num_heads_per_ct):
            base_h = h * H
            for j in range(head_dim):
                base_j = base_h + j * L
                for i in range(L):
                    t = (i + d) % L
                    mask[base_j + i] = float(A_per_head[h, i, t])
        diag_pts.append(backend._encode(mask))
    return diag_pts




def attn_synthesizer_nexus(
    backend: CKKSBackend,
    V_ct: Ciphertext,
    diag_pts: Sequence,
    *,
    L: int,
    head_dim: int,
    num_heads_per_ct: int = 1,
    cache: Optional[dict] = None,
) -> Ciphertext:
    """Synthesizer-LPAN attention: O = A·V with A as PLAINTEXT diagonals.

    Parameters
    ----------
    V_ct : NEXUS col-major ciphertext, slot[h·H + j·L + i] = V[i, j].
    diag_pts : sequence of L plaintexts, output of
        :func:`encode_synthesizer_diagonals`. Entry d is the diagonal-d
        mask; together they describe the per-head attention pattern.
    cache : dict, optional. If provided, top/bot V-cyclic-shift masks
        are encoded once and reused.

    Returns
    -------
    Single Ciphertext in NEXUS col-major layout (same shape as V_ct).

    Critical-path FHE cost per call:
      0 ct·ct  (vs L in baseline attn_apply_nexus)
      ~5L pt·ct + ~2L rotations + ~3L additions
    """
    n_slots = backend.capabilities.n_slots
    H = head_dim * L
    total = num_heads_per_ct * H
    if total > n_slots:
        raise ValueError(
            f"num_heads_per_ct·L·head_dim={total} > num_slots={n_slots}"
        )
    if len(diag_pts) != L:
        raise ValueError(f"len(diag_pts)={len(diag_pts)} != L={L}")

    n_cols = num_heads_per_ct * head_dim
    cache_key = ("synth_av_shift", L, head_dim, num_heads_per_ct)
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        top_pts = cached["top_pts"]
        bot_pts = cached["bot_pts"]
    else:
        # Identical V cyclic-shift masks as in attn_apply_nexus.
        top_pts: List = [None]
        bot_pts: List = [None]
        for d in range(1, L):
            top_mask = [0.0] * n_slots
            bot_mask = [0.0] * n_slots
            for j in range(n_cols):
                base = j * L
                for i in range(L - d):
                    top_mask[base + i] = 1.0
                for i in range(L - d, L):
                    bot_mask[base + i] = 1.0
            top_pts.append(backend._encode(top_mask))
            bot_pts.append(backend._encode(bot_mask))
        if cache is not None:
            cache[cache_key] = {"top_pts": top_pts, "bot_pts": bot_pts}

    Out: Optional[Ciphertext] = None
    for d in range(L):
        # 1. V cyclic shift along i-axis by d (within each j-column).
        if d == 0:
            # Match d>=1 depth so cached diag_pts can be uniform-depth.
            V_shift = backend._clone(V_ct)
            backend._ops.mod_drop_inplace_ct(V_shift)
        else:
            V_left  = backend.rotate(V_ct, d)
            V_right = backend.rotate(V_ct, d - L)
            V_shift = backend.add(
                backend._mul_plain_pt(V_left, top_pts[d]),
                backend._mul_plain_pt(V_right, bot_pts[d]),
            )

        # 2. multiply by diagonal-d plaintext (NO ct·ct).
        prod = backend._mul_plain_pt(V_shift, diag_pts[d])

        # 3. accumulate.
        Out = prod if Out is None else backend.add(Out, prod)

    if Out is None:
        Out = backend.mul_plain(V_ct, [0.0] * n_slots)
    return Out




def encode_synthesizer_bsgs(
    backend: CKKSBackend,
    A_per_head: np.ndarray,
    *,
    L: int,
    head_dim: int,
    num_heads_per_ct: int = 1,
) -> dict:
    """Pre-encode BSGS-fused plaintexts for :func:`attn_synthesizer_bsgs_nexus`.

    Returns a dict with two key-arrays of length L:
      - ``top_combined[d]`` = mask_top[d] · diag[d]   (plaintext)
      - ``bot_combined[d]`` = mask_bot[d] · diag[d]   (plaintext)
    plus pre-rotated giant-step versions used by the BSGS inner loop.

    The fusion of top/bot mask × diagonal turns 2L rotations + 2L mul_plain
    + L add into 2·(2√L) rotations + 2L mul_plain + 2L add — eliminating
    the dominant rotation cost while preserving exact mathematical
    equivalence to :func:`attn_synthesizer_nexus`.
    """
    n_slots = backend.capabilities.n_slots
    H = head_dim * L
    if A_per_head.shape != (num_heads_per_ct, L, L):
        raise ValueError(
            f"A_per_head shape {A_per_head.shape} != "
            f"({num_heads_per_ct}, {L}, {L})"
        )

    # BSGS dims: bs * gs = L, both ≈ √L.
    bs = 1
    while bs * bs < L:
        bs <<= 1
    if bs * bs > L:
        bs >>= 1
    gs = L // bs
    if bs * gs != L:
        bs, gs = 1, L

    def _slot_array_top_diag(d):
        out = np.zeros(n_slots, dtype=np.float64)
        for h in range(num_heads_per_ct):
            base_h = h * H
            for j in range(head_dim):
                base_j = base_h + j * L
                for i in range(L - d):
                    out[base_j + i] = A_per_head[h, i, (i + d) % L]
        return out

    def _slot_array_bot_diag(d):
        out = np.zeros(n_slots, dtype=np.float64)
        for h in range(num_heads_per_ct):
            base_h = h * H
            for j in range(head_dim):
                base_j = base_h + j * L
                for i in range(L - d, L):
                    out[base_j + i] = A_per_head[h, i, (i + d) % L]
        return out

    def _cyclic_rotate(arr, k):
        """Slot rotation matching backend.rotate convention.

        backend.rotate(ct, +k) maps slot[i] ← slot[(i+k) mod n], i.e. left
        shift by k. To pre-rotate a plaintext mask P such that
        rotate(ct·P_pre, +g·bs) yields the desired alignment, we apply the
        SAME left-shift to P inverted by g·bs amount — i.e. P_pre is P
        rotated by -g·bs (right shift by g·bs).
        """
        k %= n_slots
        if k == 0:
            return arr
        return np.concatenate([arr[k:], arr[:k]])

    top_pts_per_g = []
    bot_pts_per_g = []
    for g in range(gs):
        top_g = []
        bot_g = []
        shift = g * bs
        for b in range(bs):
            d = b + shift
            arr_top = _slot_array_top_diag(d)
            arr_bot = _slot_array_bot_diag(d)
            arr_top = _cyclic_rotate(arr_top, -shift)
            arr_bot = _cyclic_rotate(arr_bot, -shift)
            top_g.append(backend._encode(arr_top.tolist()))
            bot_g.append(backend._encode(arr_bot.tolist()))
        top_pts_per_g.append(top_g)
        bot_pts_per_g.append(bot_g)

    return {
        "bs": bs, "gs": gs, "L": L,
        "top_pts_per_g": top_pts_per_g,
        "bot_pts_per_g": bot_pts_per_g,
    }




def attn_synthesizer_bsgs_nexus(
    backend: CKKSBackend,
    V_ct: Ciphertext,
    bsgs_pts: dict,
    *,
    head_dim: int,
    num_heads_per_ct: int = 1,
) -> Ciphertext:
    """BSGS-accelerated Synthesizer-LPAN attention.

    Mathematically equivalent to :func:`attn_synthesizer_nexus`. Reduces
    rotation count from 2L = 256 (for L=128) to 2(bs+gs) ≈ 32.

    Algorithm (using d = b + g·bs):
      Σ_d rot(V, d)·C_top_d
        = Σ_g rot( Σ_b rot(V, b) · rot(C_top_{b+g·bs}, -g·bs), g·bs )
      Σ_d rot(V, d-L)·C_bot_d
        = Σ_g rot( Σ_b rot(V, b-L) · rot(C_bot_{b+g·bs}, -g·bs), g·bs )

    The plaintext rotations are precomputed in :func:`encode_synthesizer_bsgs`
    so only ciphertext rotations are charged at inference time.
    """
    L = bsgs_pts["L"]
    bs = bsgs_pts["bs"]
    gs = bsgs_pts["gs"]
    top_pts_per_g = bsgs_pts["top_pts_per_g"]
    bot_pts_per_g = bsgs_pts["bot_pts_per_g"]

    # Pre-rotate V by 0..bs-1 (positive) and by (0..bs-1) - L.
    baby_pos = [V_ct]
    for b in range(1, bs):
        baby_pos.append(backend.rotate(V_ct, b))
    baby_neg = []
    for b in range(bs):
        baby_neg.append(backend.rotate(V_ct, b - L))

    Out: Optional[Ciphertext] = None
    for g in range(gs):
        baby_sum_top: Optional[Ciphertext] = None
        for b in range(bs):
            term = backend._mul_plain_pt(baby_pos[b], top_pts_per_g[g][b])
            baby_sum_top = term if baby_sum_top is None else backend.add(baby_sum_top, term)
        top_giant = baby_sum_top if g == 0 else backend.rotate(baby_sum_top, g * bs)

        baby_sum_bot: Optional[Ciphertext] = None
        for b in range(bs):
            term = backend._mul_plain_pt(baby_neg[b], bot_pts_per_g[g][b])
            baby_sum_bot = term if baby_sum_bot is None else backend.add(baby_sum_bot, term)
        bot_giant = baby_sum_bot if g == 0 else backend.rotate(baby_sum_bot, g * bs)

        contribution = backend.add(top_giant, bot_giant)
        Out = contribution if Out is None else backend.add(Out, contribution)

    return Out


# ---------------------------------------------------------------------------
# Rotation-key pre-registration (avoids bit-decomp rotations)
# ---------------------------------------------------------------------------


