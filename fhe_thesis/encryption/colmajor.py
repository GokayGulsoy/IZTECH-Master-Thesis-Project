"""Column-major packing helpers

Sliced from the original ``ops_attention_nexus.py`` during the production
re-modularization (synthesizer-lpan-production branch).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext

# -------------------------------------------------------------------------
# Column-major packing helpers
# -------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Column-major packing helpers
# ---------------------------------------------------------------------------


def pack_colmajor(
    backend: CKKSBackend,
    X: np.ndarray,
    *,
    L: int,
    head_dim: int,
) -> Ciphertext:
    """Encrypt ``X[L, head_dim]`` in column-major layout.

    slot[j*L + i] = X[i, j]

    L must be a power of 2; pad input with zero rows if not.
    """
    if X.shape != (L, head_dim):
        raise ValueError(f"X.shape {X.shape} != ({L}, {head_dim})")
    if (L & (L - 1)) != 0:
        raise ValueError(f"L must be power of 2; got {L}")
    n_slots = backend.capabilities.n_slots
    if L * head_dim > n_slots:
        raise ValueError(f"L*head_dim={L*head_dim} > num_slots={n_slots}")
    buf = [0.0] * n_slots
    for j in range(head_dim):
        base = j * L
        for i in range(L):
            buf[base + i] = float(X[i, j])
    return backend.encrypt(buf)




def unpack_colmajor(
    backend: CKKSBackend,
    ct: Ciphertext,
    *,
    L: int,
    head_dim: int,
) -> np.ndarray:
    """Inverse of :func:`pack_colmajor`."""
    slots = backend.decrypt(ct)
    out = np.zeros((L, head_dim), dtype=np.float64)
    for j in range(head_dim):
        base = j * L
        for i in range(L):
            out[i, j] = slots[base + i]
    return out


# ---------------------------------------------------------------------------
# NEXUS-style diagonal Q@K^T
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Phase 8k: multi-ciphertext col-major (for L=128 hidden=768 at N=2^16)
# ---------------------------------------------------------------------------
#
# Layout: hidden dim of size H is split into n_cts groups of cols_per_ct
# columns each, where cols_per_ct = num_slots // L.
# x_cts[k] holds slot[j*L + i] = X[i, k*cols_per_ct + j]  for j ∈ [0, cols_per_ct).
# The last ct may have fewer than cols_per_ct active columns; pad slots
# are kept zero.


def _cols_per_ct(backend: CKKSBackend, L: int) -> int:
    return backend.capabilities.n_slots // L




def pack_colmajor_multi(
    backend: CKKSBackend,
    X: np.ndarray,
    *,
    L: int,
    hidden: int,
) -> List[Ciphertext]:
    """Encrypt ``X[L, hidden]`` as a list of column-major cts.

    Each ct holds up to ``cols_per_ct = n_slots // L`` columns.
    """
    if X.shape != (L, hidden):
        raise ValueError(f"X.shape {X.shape} != ({L}, {hidden})")
    if (L & (L - 1)) != 0:
        raise ValueError(f"L must be power of 2; got {L}")
    n_slots = backend.capabilities.n_slots
    cols_per_ct = _cols_per_ct(backend, L)
    n_cts = (hidden + cols_per_ct - 1) // cols_per_ct
    out = []
    for k in range(n_cts):
        lo = k * cols_per_ct
        hi = min(lo + cols_per_ct, hidden)
        buf = [0.0] * n_slots
        for j in range(lo, hi):
            base = (j - lo) * L
            for i in range(L):
                buf[base + i] = float(X[i, j])
        out.append(backend.encrypt(buf))
    return out




def unpack_colmajor_multi(
    backend: CKKSBackend,
    cts: Sequence[Ciphertext],
    *,
    L: int,
    hidden: int,
) -> np.ndarray:
    cols_per_ct = _cols_per_ct(backend, L)
    out = np.zeros((L, hidden), dtype=np.float64)
    for k, ct in enumerate(cts):
        slots = backend.decrypt(ct)
        lo = k * cols_per_ct
        hi = min(lo + cols_per_ct, hidden)
        for j in range(lo, hi):
            base = (j - lo) * L
            for i in range(L):
                out[i, j] = slots[base + i]
    return out




# ---------------------------------------------------------------------------
# Rotation-key pre-registration (avoids bit-decomp rotations)
# ---------------------------------------------------------------------------


def prepare_colmajor_keys(
    backend: CKKSBackend,
    *,
    L: int,
    max_dim: int,
) -> int:
    """Pre-register every rotation step needed by the col-major kernels.

    With BSGS in ``linear_colmajor`` we only need ``bs + gs ≈ 2·sqrt(max_dim)``
    distinct shifts (positive and negative) instead of ``max_dim`` of them.
    Also covers ``per_col_sum_then_broadcast`` which uses ±L, ±2L, ±4L, ...

    For BERT-base hidden_padded=1024 → bs=gs=32 → 62 keys × ~32 MB ≈ 2 GB
    (vs 1023 keys × 32 MB ≈ 33 GB in the naive variant — which OOMs on H100).

    Returns the number of *new* shifts registered.
    """
    if not hasattr(backend, "register_rotation_keys"):
        return 0
    n = 1
    while n < max_dim:
        n <<= 1
    log2_n = n.bit_length() - 1
    bs = 1 << (log2_n // 2)
    gs = n // bs
    shifts = set()
    # BSGS baby/giant for linear_colmajor
    for i in range(1, bs):
        shifts.add(i * L)
        shifts.add(-i * L)
    for j in range(1, gs):
        shifts.add(j * bs * L)
        shifts.add(-j * bs * L)
    # per_col_sum_then_broadcast uses ±L, ±2L, ±4L, ..., ±n/2 · L
    s = L
    while s < n * L:
        shifts.add(s)
        shifts.add(-s)
        s <<= 1
    return backend.register_rotation_keys(sorted(shifts))


