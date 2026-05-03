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

Cost model (BERT-Base linear, ``hidden_dim = in_dim = out_dim = 768``,
``D = 1024`` block, ``B = 4`` tokens per ciphertext):

  ============================== =================== ===================
                                  TokenPackedTensor   MatrixPackedTensor
  ============================== =================== ===================
  rotations per token (matmul)         n   = 1024            2n / B = 512
  mul_plain per token  (matmul)        n   = 1024            2n / B = 512
  ============================== =================== ===================

The matrix layout therefore halves the rotation count per token at the
matmul level, on top of the 4× speedup it already brings to the
slot-local operators that dominate ~30 % of LPAN-FHE wall time.
"""

from __future__ import annotations

import hashlib
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import CKKSBackend, Ciphertext
from .matrix_packing import MatrixPackedTensor, next_pow2


# Cache of (Halevi-Shoup diagonals, block-mask pair) keyed by
# (sha1(weight bytes), block, num_slots). Building these is several ms of
# Python work and must not happen on every token of every layer.
_DiagCacheEntry = Tuple[List[Optional[List[float]]], int]
_diag_cache: Dict[str, _DiagCacheEntry] = {}
_mask_cache: Dict[Tuple[int, int, int], Tuple[List[float], List[float]]] = {}
_cache_lock = Lock()


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
    for ct in mpt.cts:
        # Step 1: replicate input within each block so the cyclic
        # diagonal-rotation accesses are well-defined.
        x = _replicate_in_block(
            backend, ct, in_dim=in_dim, block=block, num_slots=num_slots
        )

        # Step 2: Halevi-Shoup multiply-accumulate.
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
