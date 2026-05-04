"""Matrix-packed CKKS kernels (block-aware Halevi-Shoup linear layer).

Companion to :mod:`matrix_packing`. While the slot-local ops on a
:class:`MatrixPackedTensor` are trivial (one ``mul_plain`` / ``polyval``
processes ``B`` tokens at once), the *rotation-bearing* ops — matmul,
``sum_slots``, attention scores — must respect the per-token block
boundary. A naive cyclic rotation of the whole ciphertext blends data
from neighbouring tokens.

This module provides the primitives that make those ops correct:

* :func:`per_block_rotate_left` — masked-rotate trick that performs an
  *intra-block* cyclic rotation by ``shift`` slots, leaving every other
  block's data put. Cost: ``2`` rotations + ``2`` ``mul_plain`` per call.
* :func:`enc_linear_matrix` — Halevi-Shoup diagonal matmul applied
  per-block, so a single matmul call processes the ``B`` tokens stored
  in each input ciphertext at once.
* :func:`per_block_sum` — reduces the first ``hidden_dim`` slots of each
  block to a broadcast scalar within the block (used by LN/softmax).
* :func:`enc_gelu_matrix`, :func:`enc_layernorm_matrix`,
  :func:`enc_softmax_matrix` — slot-local LPAN polynomials applied to
  every packed token simultaneously.
"""

from __future__ import annotations

import hashlib
from math import comb
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import CKKSBackend, Ciphertext
from .coefficients import PolyCoeffs
from .matrix_packing import MatrixPackedTensor, next_pow2


_DiagCacheEntry = Tuple[List[Optional[List[float]]], int]
_diag_cache: Dict[str, _DiagCacheEntry] = {}
_mask_cache: Dict[Tuple[int, int, int], Tuple[List[float], List[float]]] = {}
_cache_lock = Lock()


# ---------------------------------------------------------------------------
# Affine substitution helper (mirrors ops._absorb_affine)
# ---------------------------------------------------------------------------


def _absorb_affine(
    power_coeffs: Sequence[float], scale: float, shift: float
) -> List[float]:
    c = list(power_coeffs)
    d = len(c) - 1
    out = [0.0] * (d + 1)
    for k in range(d + 1):
        acc = 0.0
        sk = scale**k
        for i in range(k, d + 1):
            acc += c[i] * comb(i, k) * sk * (shift ** (i - k))
        out[k] = acc
    return out


# ---------------------------------------------------------------------------
# Block-aware primitives
# ---------------------------------------------------------------------------


def _block_masks(
    block: int, shift: int, num_slots: int
) -> Tuple[List[float], List[float]]:
    """Return the (low, high) mask pair used by :func:`per_block_rotate_left`.

    ``low``  is 1.0 in the first ``block - shift`` slots of every block,
    ``high`` is 1.0 in the trailing ``shift`` slots of every block.
    Memoised because the same mask pair is reused by every diagonal of
    every matmul of every layer.
    """
    key = (block, shift, num_slots)
    with _cache_lock:
        cached = _mask_cache.get(key)
        if cached is not None:
            return cached
    if not 0 < shift < block:
        raise ValueError(f"shift must be in (0, block); got shift={shift}, block={block}")
    if num_slots % block != 0:
        raise ValueError(f"num_slots={num_slots} is not a multiple of block={block}")

    B = num_slots // block
    low = [0.0] * num_slots
    high = [0.0] * num_slots
    for b in range(B):
        base = b * block
        for j in range(block - shift):
            low[base + j] = 1.0
        for j in range(block - shift, block):
            high[base + j] = 1.0
    with _cache_lock:
        _mask_cache[key] = (low, high)
    return low, high


def per_block_rotate_left(
    backend: CKKSBackend,
    ct: Ciphertext,
    shift: int,
    *,
    block: int,
    num_slots: int,
) -> Ciphertext:
    """Cyclically rotate every ``block``-slot block left by ``shift``.

    Implementation: we perform two whole-ciphertext rotations (one left
    by ``shift``, one right by ``block - shift``) and combine them with
    complementary block-aligned masks so the data of each block stays
    inside that block.
    """
    s = shift % block
    if s == 0:
        return ct
    low, high = _block_masks(block, s, num_slots)
    rot_left = backend.rotate(ct, s)
    # ``rotate(ct, s - block)`` is a right-rotation by ``block - s``.
    rot_right = backend.rotate(ct, s - block)
    return backend.add(
        backend.mul_plain(rot_left, low),
        backend.mul_plain(rot_right, high),
    )


def _replicate_in_block(
    backend: CKKSBackend,
    ct: Ciphertext,
    *,
    in_dim: int,
    block: int,
    num_slots: int,
) -> Ciphertext:
    """Cyclically replicate the first ``in_dim`` slots of every block to fill the block.

    Halevi-Shoup matmul assumes the input is *cyclically* replicated to
    length ``n``. Our blocks already have ``in_dim`` real values
    followed by ``block - in_dim`` zeros; doubling-and-adding via
    per-block rotation populates the whole block.
    """
    cur = next_pow2(in_dim)
    if cur == 0:
        cur = 1
    out = ct
    while cur < block:
        shifted = per_block_rotate_left(
            backend, out, block - cur, block=block, num_slots=num_slots
        )
        # rotating left by (block - cur) puts old slot 0 at slot (cur),
        # i.e. data block [0:cur) lands at [cur:2cur). Adding completes
        # one round of cyclic-replication doubling.
        out = backend.add(out, shifted)
        cur <<= 1
    return out


# ---------------------------------------------------------------------------
# Halevi-Shoup matmul (per-block)
# ---------------------------------------------------------------------------


def _diagonals_for_weight(
    weight: np.ndarray, n: int, block: int, num_slots: int
) -> List[Optional[List[float]]]:
    """Build the ``n`` Halevi-Shoup diagonals tiled across every block.

    For diagonal index ``i ∈ [0, n)``::

        diag_i[k] = W[k, (k + i) mod n]    (k ∈ [0, out_dim), within one block)

    The same length-``n`` pattern is replicated in every B-block so that
    a single ``mul_plain`` lands the diagonal on all packed tokens at
    once.
    """
    out_dim, in_dim = weight.shape
    B = num_slots // block
    diagonals: List[Optional[List[float]]] = []
    for i in range(n):
        diag = [0.0] * num_slots
        any_nz = False
        for k in range(out_dim):
            col = (k + i) % n
            if col < in_dim:
                val = float(weight[k, col])
                if val != 0.0:
                    any_nz = True
                # Tile across all B blocks.
                for b in range(B):
                    diag[b * block + k] = val
        diagonals.append(diag if any_nz else None)
    return diagonals


def enc_linear_matrix(
    backend: CKKSBackend,
    mpt: MatrixPackedTensor,
    weight: Sequence[Sequence[float]],
    bias: Optional[Sequence[float]] = None,
) -> MatrixPackedTensor:
    """Compute ``Y = X · Wᵀ + b`` on a matrix-packed tensor.

    ``weight`` has shape ``(out_dim, in_dim)`` row-major. The output
    keeps the same packing layout as the input (``B`` tokens per
    ciphertext, stride ``mpt.block``); the per-token logical width
    becomes ``out_dim``.
    """
    w_arr = np.asarray(weight, dtype=np.float64)
    if w_arr.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {w_arr.shape}")
    out_dim, in_dim = w_arr.shape
    if in_dim != mpt.hidden_dim:
        raise ValueError(
            f"weight in_dim={in_dim} != mpt.hidden_dim={mpt.hidden_dim}"
        )

    block = mpt.block
    num_slots = mpt.num_slots
    n = max(next_pow2(out_dim), next_pow2(in_dim))
    if n > block:
        raise ValueError(
            f"matmul dim n={n} exceeds packing block={block}; "
            f"re-encrypt with a larger block (e.g. via MatrixPackedTensor.encrypt(..., block={n}))"
        )

    cache_key = hashlib.sha1(
        w_arr.tobytes() + f"|n={n}|block={block}|slots={num_slots}".encode()
    ).hexdigest()
    with _cache_lock:
        cached = _diag_cache.get(cache_key)
    if cached is None:
        diagonals = _diagonals_for_weight(w_arr, n, block, num_slots)
        with _cache_lock:
            _diag_cache[cache_key] = (diagonals, n)
    else:
        diagonals, n = cached

    # Per-block-replicated bias (length num_slots).
    bias_vec: Optional[List[float]] = None
    if bias is not None:
        b_list = list(bias)
        if len(b_list) != out_dim:
            raise ValueError(f"bias length {len(b_list)} != out_dim {out_dim}")
        bias_vec = [0.0] * num_slots
        B = num_slots // block
        for b in range(B):
            base = b * block
            for j, v in enumerate(b_list):
                bias_vec[base + j] = float(v)

    out_cts: List[Ciphertext] = []
    fast = hasattr(backend, "halevi_shoup_matmul")

    for ct in mpt.cts:
        # Step 1: replicate input within each block so the cyclic
        # diagonal-rotation accesses are well-defined.
        x = _replicate_in_block(
            backend, ct, in_dim=in_dim, block=block, num_slots=num_slots
        )

        if fast:
            # Fast path: one C++ call for the entire diagonal sweep.
            shifts = list(range(n))
            low_masks: List[Optional[List[float]]] = [None] * n
            high_masks: List[Optional[List[float]]] = [None] * n
            for i in range(1, n):
                lo, hi = _block_masks(block, i, num_slots)
                low_masks[i] = lo
                high_masks[i] = hi
            result_ct = backend.halevi_shoup_matmul(
                x,
                block=block,
                shifts=shifts,
                diagonals=diagonals,
                low_masks=low_masks,
                high_masks=high_masks,
                bias_vec=bias_vec,
            )
            out_cts.append(result_ct)
            continue

        # Slow path (fallback): Python loop over diagonals.
        result: Optional[Ciphertext] = None
        for i, diag in enumerate(diagonals):
            if diag is None:
                continue
            rot_x = (
                x
                if i == 0
                else per_block_rotate_left(
                    backend, x, i, block=block, num_slots=num_slots
                )
            )
            term = backend.mul_plain(rot_x, diag)
            result = term if result is None else backend.add(result, term)

        if result is None:
            result = backend.mul_plain(ct, [0.0] * num_slots)
        if bias_vec is not None:
            result = backend.add_plain(result, bias_vec)
        out_cts.append(result)

    return MatrixPackedTensor.from_ciphertexts(
        out_cts,
        seq_len=mpt.seq_len,
        hidden_dim=out_dim,
        block=block,
        tokens_per_ct=mpt.tokens_per_ct,
        num_slots=num_slots,
    )


# ---------------------------------------------------------------------------
# Per-block sum reduction (used by LN and softmax)
# ---------------------------------------------------------------------------


def per_block_sum(
    backend: CKKSBackend,
    ct: Ciphertext,
    *,
    hidden_dim: int,
    block: int,
    num_slots: int,
) -> Ciphertext:
    """Sum the first ``hidden_dim`` slots of every block; broadcast within block.

    Operates entirely inside each ``block``-slot region — the result has
    the per-block sum replicated across all ``block`` slots of that
    block, exactly mirroring the global ``sum_slots`` semantics but
    constrained to the block.

    Cost: ``log2(block)`` per-block rotations + adds (≈ ``2 log2(block)``
    raw rotations because each per-block rotation = 2 full rotations).

    Assumes the trailing ``block - hidden_dim`` slots are zero (this is
    the contract of :class:`MatrixPackedTensor`); otherwise the sum
    will include garbage from the pad.
    """
    out = ct
    step = 1
    while step < block:
        shifted = per_block_rotate_left(
            backend, out, step, block=block, num_slots=num_slots
        )
        out = backend.add(out, shifted)
        step <<= 1
    return out


# ---------------------------------------------------------------------------
# Slot-local LPAN polynomials — apply to all packed tokens at once
# ---------------------------------------------------------------------------


def _apply_slot_local_poly(
    backend: CKKSBackend,
    mpt: MatrixPackedTensor,
    absorbed: Sequence[float],
    *,
    keep_pad_zero: bool = True,
) -> MatrixPackedTensor:
    """Run ``polyval(absorbed, ·)`` on every ciphertext of ``mpt``.

    Polynomials with non-zero constant term will lift the trailing
    ``block - hidden_dim`` pad slots from zero to ``c_0``; if any
    downstream op assumes the pad is zero (LN sum, softmax sum) we
    must re-zero with the block mask. Cost: 1 mul_plain per ct.
    """
    out_cts: List[Ciphertext] = []
    mask = mpt.block_mask(mpt.hidden_dim) if keep_pad_zero else None
    for ct in mpt.cts:
        y = backend.polyval(ct, list(absorbed))
        if mask is not None:
            y = backend.mul_plain(y, mask)
        out_cts.append(y)
    return MatrixPackedTensor.from_ciphertexts(
        out_cts,
        seq_len=mpt.seq_len,
        hidden_dim=mpt.hidden_dim,
        block=mpt.block,
        tokens_per_ct=mpt.tokens_per_ct,
        num_slots=mpt.num_slots,
    )


def enc_gelu_matrix(
    backend: CKKSBackend,
    x: MatrixPackedTensor,
    power_coeffs: Sequence[float],
    interval: Tuple[float, float],
) -> MatrixPackedTensor:
    """Element-wise GELU polynomial across all packed tokens.

    Folds the ``[a,b] → [-1,1]`` standardisation into the coefficients
    via :func:`_absorb_affine` so it consumes zero CKKS levels.
    """
    a, b = interval
    s = 2.0 / (b - a)
    sh = -(a + b) / (b - a)
    absorbed = _absorb_affine(power_coeffs, s, sh)
    # GELU is fed directly into the next linear, which uses Halevi-Shoup
    # cyclic replication — non-zero pad would corrupt that. Mask it.
    return _apply_slot_local_poly(backend, x, absorbed, keep_pad_zero=True)


def enc_layernorm_matrix(
    backend: CKKSBackend,
    x: MatrixPackedTensor,
    invsqrt_power_coeffs: Sequence[float],
    invsqrt_interval: Tuple[float, float],
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> MatrixPackedTensor:
    """LPAN LayerNorm on all packed tokens at once.

    Identical math to :func:`fhe_thesis.encryption.ops.enc_ln_poly` but
    every per-token reduction is done with :func:`per_block_sum` instead
    of a global ``sum_slots``, so all ``B`` tokens in a ciphertext are
    normalised independently and in parallel.
    """
    h = x.hidden_dim
    inv_h = 1.0 / h
    a, b = invsqrt_interval
    sscale = 2.0 / (b - a)
    sshift = -(a + b) / (b - a)
    absorbed_invsqrt = _absorb_affine(invsqrt_power_coeffs, sscale, sshift)

    block = x.block
    num_slots = x.num_slots
    bmask = x.block_mask(h)
    # gamma/beta replicated across the B blocks so the same per-feature
    # scale/shift lands on every packed token.
    gamma_rep = x.replicated_vector(gamma.tolist())
    beta_rep = x.replicated_vector(beta.tolist())
    # 1/h scale folded into the mask used for mean recovery.
    mean_mask = [v * inv_h for v in bmask]

    out_cts: List[Ciphertext] = []
    for ct in x.cts:
        # μ_block = sum_block(x) / h, broadcast to first h slots of block.
        s_ct = per_block_sum(
            backend, ct, hidden_dim=h, block=block, num_slots=num_slots
        )
        mean_bc = backend.mul_plain(s_ct, mean_mask)
        centred = backend.sub(ct, mean_bc)
        # Re-zero the pad after subtraction (mean_bc was masked but
        # `ct` had zeros there; subtraction keeps them zero).
        sq = backend.mul(centred, centred)
        # Variance = sum_block(sq) / h, broadcast.
        s_sq = per_block_sum(
            backend, sq, hidden_dim=h, block=block, num_slots=num_slots
        )
        var_bc = backend.mul_plain(s_sq, mean_mask)
        inv_sigma = backend.polyval(var_bc, list(absorbed_invsqrt))
        # γ/σ · (x − μ) + β
        scaled = backend.mul(centred, inv_sigma)
        scaled = backend.mul_plain(scaled, gamma_rep)
        out_cts.append(backend.add_plain(scaled, beta_rep))

    return MatrixPackedTensor.from_ciphertexts(
        out_cts,
        seq_len=x.seq_len,
        hidden_dim=h,
        block=block,
        tokens_per_ct=x.tokens_per_ct,
        num_slots=num_slots,
    )


def enc_softmax_matrix(
    backend: CKKSBackend,
    scores: MatrixPackedTensor,
    power_coeffs: Sequence[float],
    interval: Tuple[float, float],
) -> MatrixPackedTensor:
    """Element-wise LPAN softmax polynomial across all packed score rows.

    Identical to :func:`enc_gelu_matrix` semantically — softmax is
    applied as a single power-basis polynomial on each slot. The
    division by Σ exp is performed elsewhere (caller fuses into
    :func:`enc_attention_apply_matrix` or precomputes via the LPAN
    coefficient tables).
    """
    a, b = interval
    s = 2.0 / (b - a)
    sh = -(a + b) / (b - a)
    absorbed = _absorb_affine(power_coeffs, s, sh)
    return _apply_slot_local_poly(backend, scores, absorbed, keep_pad_zero=True)


# ---------------------------------------------------------------------------
# Attention kernels
# ---------------------------------------------------------------------------
#
# Implementation notes — these are MVP ports that preserve the per-token
# semantics of the token-packed kernels in :mod:`ops`. Q, K, V are kept
# in matrix-packed layout so the surrounding linear projections keep
# their B× speedup; inside the attention dot-products we *extract* one
# token at a time from each matrix-packed ct (one rotate + one mask)
# and run the existing per-token loop. This is correct and reuses
# every primitive (`dot`, `sum_slots`, `place_scaled_at_slot`) we have
# already validated, but **does not give attention the same B× speedup
# as the linears** — the L² inner loop is unchanged. Fully diagonal
# attention (à la NEXUS) is the next optimisation pass once the rest of
# the matrix pipeline is end-to-end stable.


def _extract_token(
    backend: CKKSBackend, mpt: MatrixPackedTensor, idx: int
) -> Ciphertext:
    """Return a ciphertext whose first ``hidden_dim`` slots hold token ``idx``.

    Token ``idx`` lives in ciphertext ``g = idx // B`` at block
    ``k = idx % B`` (slots ``[k·D, k·D + hidden_dim)``). We rotate left
    by ``k·D`` to land it at slot 0, then mask the first ``hidden_dim``
    slots to zero out neighbouring tokens.
    """
    g, k = divmod(idx, mpt.tokens_per_ct)
    ct = mpt.cts[g]
    if k > 0:
        ct = backend.rotate(ct, k * mpt.block)
    mask = [0.0] * mpt.num_slots
    for j in range(mpt.hidden_dim):
        mask[j] = 1.0
    return backend.mul_plain(ct, mask)


def enc_qk_scores_matrix(
    backend: CKKSBackend,
    Q: MatrixPackedTensor,
    K: MatrixPackedTensor,
    scale: float,
) -> MatrixPackedTensor:
    """Compute ``S[i, j] = scale · ⟨Q[i], K[j]⟩`` matrix-packed.

    Output layout: ``(seq_len, seq_len)`` matrix-packed with stride
    ``D_out = next_pow2(seq_len)`` so the result flows directly into
    :func:`enc_softmax_matrix` and :func:`enc_attention_apply_matrix`.
    """
    if Q.seq_len != K.seq_len:
        raise ValueError(f"Q.seq_len {Q.seq_len} != K.seq_len {K.seq_len}")
    if Q.hidden_dim != K.hidden_dim:
        raise ValueError(
            f"Q.hidden_dim {Q.hidden_dim} != K.hidden_dim {K.hidden_dim}"
        )

    L = Q.seq_len
    num_slots = Q.num_slots
    D_out = next_pow2(L)
    if D_out > num_slots:
        raise ValueError(
            f"seq_len {L} too large: D_out={D_out} > num_slots={num_slots}"
        )
    B_out = num_slots // D_out
    n_out_cts = (L + B_out - 1) // B_out

    # Pre-extract every key token once (reused for every query row).
    keys = [_extract_token(backend, K, j) for j in range(L)]

    out_cts: List[Optional[Ciphertext]] = [None] * n_out_cts
    for i in range(L):
        q_i = _extract_token(backend, Q, i)
        g, k = divmod(i, B_out)
        base = k * D_out
        row_acc: Optional[Ciphertext] = None
        for j in range(L):
            s = backend.dot(q_i, keys[j])
            term = backend.place_scaled_at_slot(s, base + j, num_slots, scale)
            row_acc = term if row_acc is None else backend.add(row_acc, term)
        if out_cts[g] is None:
            out_cts[g] = row_acc
        else:
            out_cts[g] = backend.add(out_cts[g], row_acc)

    # Materialise any unused output ct as zero (shouldn't happen for L>0).
    final_cts: List[Ciphertext] = []
    for ct in out_cts:
        if ct is None:
            ct = backend.mul_plain(Q.cts[0], [0.0] * num_slots)
        final_cts.append(ct)

    return MatrixPackedTensor.from_ciphertexts(
        final_cts,
        seq_len=L,
        hidden_dim=L,
        block=D_out,
        tokens_per_ct=B_out,
        num_slots=num_slots,
    )


def enc_attention_apply_matrix(
    backend: CKKSBackend,
    attn: MatrixPackedTensor,
    V: MatrixPackedTensor,
) -> MatrixPackedTensor:
    """Compute ``out[i] = Σ_j attn[i, j] · V[j]`` matrix-packed.

    ``attn`` has shape ``(L, L)`` (output of :func:`enc_softmax_matrix`),
    ``V`` has shape ``(L, head_dim)``. Output keeps ``V``'s packing
    layout so it flows straight into the next linear projection.
    """
    if attn.seq_len != V.seq_len:
        raise ValueError(
            f"attn.seq_len {attn.seq_len} != V.seq_len {V.seq_len}"
        )
    if attn.hidden_dim != attn.seq_len:
        raise ValueError(
            "attn must be square (seq_len, seq_len); "
            f"got ({attn.seq_len}, {attn.hidden_dim})"
        )

    L = attn.seq_len
    head_dim = V.hidden_dim
    num_slots = V.num_slots
    D_out = V.block
    B_out = V.tokens_per_ct
    n_out_cts = len(V.cts)

    # Pre-extract V token rows once.
    vs = [_extract_token(backend, V, j) for j in range(L)]

    # One-hot masks for picking attn[i, j] out of the L-slot row.
    masks: List[List[float]] = []
    for j in range(L):
        m = [0.0] * num_slots
        m[j] = 1.0
        masks.append(m)

    out_cts: List[Optional[Ciphertext]] = [None] * n_out_cts
    for i in range(L):
        attn_i = _extract_token(backend, attn, i)  # values in slots [0..L)
        row: Optional[Ciphertext] = None
        for j in range(L):
            slot = backend.mul_plain(attn_i, masks[j])
            scalar = backend.sum_slots(slot)
            scalar_bcast = backend.broadcast_first_slot(scalar, head_dim, 1.0)
            term = backend.mul(vs[j], scalar_bcast)
            row = term if row is None else backend.add(row, term)
        # Place ``row`` (values in slots [0..head_dim)) at the correct
        # block in the output ciphertext for token i.
        g, k = divmod(i, B_out)
        if k > 0:
            # right-rotate by k·D_out  (== rotate by negative offset)
            row = backend.rotate(row, -k * D_out)
        if out_cts[g] is None:
            out_cts[g] = row
        else:
            out_cts[g] = backend.add(out_cts[g], row)

    final_cts: List[Ciphertext] = []
    for ct in out_cts:
        if ct is None:
            ct = backend.mul_plain(V.cts[0], [0.0] * num_slots)
        final_cts.append(ct)

    return MatrixPackedTensor.from_ciphertexts(
        final_cts,
        seq_len=L,
        hidden_dim=head_dim,
        block=D_out,
        tokens_per_ct=B_out,
        num_slots=num_slots,
    )


# ---------------------------------------------------------------------------
# Multi-head self-attention (matrix-packed)
# ---------------------------------------------------------------------------


def _concat_heads_matrix(
    backend: CKKSBackend,
    heads: List[MatrixPackedTensor],
    hidden_dim: int,
    target_block: int,
) -> MatrixPackedTensor:
    """Concatenate per-head matrix-packed outputs into one MPT of width hidden_dim.

    Each head's MPT has its values in slots ``[0..head_dim)`` of every
    block (with the trailing ``block - head_dim`` slots zero). To produce
    the concatenated output we right-shift head ``h`` *within each block*
    by ``h·head_dim`` so its data lands at slots
    ``[h·head_dim, (h+1)·head_dim)`` of the block, then add.

    Pre-condition: every head's MPT must use the same ``block``,
    ``tokens_per_ct``, ``num_slots`` and ``seq_len``; we assert this.
    """
    if not heads:
        raise ValueError("no heads to concat")
    head_dim = heads[0].hidden_dim
    block = heads[0].block
    num_slots = heads[0].num_slots
    B = heads[0].tokens_per_ct
    L = heads[0].seq_len
    H = len(heads)
    if head_dim * H != hidden_dim:
        raise ValueError(
            f"head_dim·H ({head_dim}·{H}) != hidden_dim {hidden_dim}"
        )
    if block != target_block:
        raise ValueError(
            f"per-head block {block} != target_block {target_block}; "
            "project Q/K/V with block=hidden_dim_full to enable in-block concat"
        )

    n_cts = len(heads[0].cts)
    out_cts: List[Ciphertext] = []
    for i in range(n_cts):
        row = heads[0].cts[i]
        for h in range(1, H):
            shift = h * head_dim
            # right-rotate each block by `shift` slots → equivalent to
            # per_block_rotate_left by (block - shift).
            shifted = per_block_rotate_left(
                backend, heads[h].cts[i], block - shift,
                block=block, num_slots=num_slots,
            )
            row = backend.add(row, shifted)
        out_cts.append(row)

    return MatrixPackedTensor.from_ciphertexts(
        out_cts,
        seq_len=L,
        hidden_dim=hidden_dim,
        block=block,
        tokens_per_ct=B,
        num_slots=num_slots,
    )


def enc_self_attention_matrix(
    backend: CKKSBackend,
    x: MatrixPackedTensor,
    Wq: np.ndarray, bq: np.ndarray,
    Wk: np.ndarray, bk: np.ndarray,
    Wv: np.ndarray, bv: np.ndarray,
    Wo: np.ndarray, bo: np.ndarray,
    softmax_coeffs: "PolyCoeffs",
    num_heads: int,
) -> MatrixPackedTensor:
    """Multi-head self-attention on a matrix-packed activation.

    Mirrors :func:`fhe_thesis.encryption.ops.enc_self_attention` but
    every primitive is the matrix-packed variant. The output keeps the
    input's packing layout (same ``block`` / ``tokens_per_ct``) so it
    flows directly into the residual + LN.
    """
    if Wq.shape != Wk.shape or Wq.shape != Wv.shape:
        raise ValueError("Q/K/V weight shapes must match")
    hidden = Wq.shape[0]
    if hidden % num_heads != 0:
        raise ValueError(f"hidden {hidden} not divisible by num_heads {num_heads}")
    head_dim = hidden // num_heads
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)
    target_block = x.block  # keep per-head block = full block for in-block concat

    head_outputs: List[MatrixPackedTensor] = []
    for h in range(num_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        # Per-head Q/K/V projections at *full* block stride so the
        # values live in slots [0..head_dim) of every D_full-wide block.
        Q_h = enc_linear_matrix(backend, x, Wq[s:e, :], bias=bq[s:e])
        K_h = enc_linear_matrix(backend, x, Wk[s:e, :], bias=bk[s:e])
        V_h = enc_linear_matrix(backend, x, Wv[s:e, :], bias=bv[s:e])

        # Score: (L, head_dim) × (L, head_dim)ᵀ → (L, L) matrix-packed.
        S = enc_qk_scores_matrix(backend, Q_h, K_h, scale=inv_sqrt_d)

        # Softmax polynomial (slot-local).
        if softmax_coeffs.per_head:
            head_power_coeffs = softmax_coeffs.power_coeffs[h].tolist()
        else:
            head_power_coeffs = softmax_coeffs.power_coeffs.tolist()
        A = enc_softmax_matrix(
            backend, S, head_power_coeffs, softmax_coeffs.interval
        )

        # attn @ V → (L, head_dim) matrix-packed (block = V_h.block).
        head_outputs.append(enc_attention_apply_matrix(backend, A, V_h))

    concat = _concat_heads_matrix(backend, head_outputs, hidden, target_block)
    return enc_linear_matrix(backend, concat, Wo, bias=bo)
