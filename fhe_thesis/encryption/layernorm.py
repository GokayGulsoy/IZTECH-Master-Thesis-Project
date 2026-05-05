"""Column-major LayerNorm (single-ct + multi-bundle)

Sliced from the original ``ops_attention_nexus.py`` during the production
re-modularization (synthesizer-lpan-production branch).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext
from .colmajor import _cols_per_ct  # noqa: F401
from .multi import sub_multi, mul_multi, per_col_sum_multi  # noqa: F401

# -------------------------------------------------------------------------
# Column-major LayerNorm (single-ct + multi-bundle)
# -------------------------------------------------------------------------



def layernorm_colmajor(
    backend: CKKSBackend,
    x_ct: Ciphertext,
    *,
    L: int,
    hidden_dim: int,
    invsqrt_power_coeffs: Sequence[float],
    invsqrt_interval: tuple,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> Ciphertext:
    """LPAN polynomial LayerNorm in the column-major slot layout.

    For each row i ∈ [0, L):
        μ_i  = (1/h) Σ_j X[i, j]
        σ²_i = (1/h) Σ_j (X[i, j] − μ_i)²
        Y[i, j] = γ_j · (X[i, j] − μ_i) · poly_invsqrt(σ²_i + eps) + β_j

    The eps is folded into ``invsqrt_interval`` by the caller (or is
    implicitly zero for the trained polynomial). γ/β are per-feature
    (per-column) constants.

    Depth: 1 (mean) + 1 (square) + 1 (var mul_plain) + invsqrt_poly_depth
    + 2 (gamma·(x-μ)·invσ) ≈ matches the row-major version.
    """
    h = hidden_dim
    inv_h = 1.0 / h
    n_slots = backend.capabilities.n_slots

    # Internal padding to next power of 2 (per_col_sum_then_broadcast handles
    # this transparently). Pad slots are assumed zero on input.
    h_pad = 1
    while h_pad < h:
        h_pad <<= 1

    a, b = invsqrt_interval
    sscale = 2.0 / (b - a)
    sshift = -(a + b) / (b - a)
    # Affine absorb: poly evaluated on (s·x + sh) instead of x.
    n = len(invsqrt_power_coeffs)
    absorbed = [0.0] * n
    # (s·x + sh)^k  expanded via binomial.
    from math import comb
    for k, c in enumerate(invsqrt_power_coeffs):
        for r in range(k + 1):
            absorbed[r] += float(c) * (sscale ** r) * (sshift ** (k - r)) * comb(k, r)

    # γ / β replicated col-major: slot[j*L + i] = γ[j] / β[j].
    gamma_vec = [0.0] * n_slots
    beta_vec = [0.0] * n_slots
    for j in range(h):
        base = j * L
        for i in range(L):
            gamma_vec[base + i] = float(gamma[j])
            beta_vec[base + i] = float(beta[j])

    # μ broadcast.
    mean_bc = per_col_sum_then_broadcast(
        backend, x_ct, L=L, hidden_dim=h, num_slots=n_slots, scale=inv_h,
    )
    centred = backend.sub(x_ct, mean_bc)
    # When h is not a power of 2, mean_bc spans the padded region too, so
    # centred has -μ in pad slots. Mask them off before squaring so the
    # variance sum is correct. (No-op when h == h_pad.)
    if h_pad != h:
        active_mask = [0.0] * n_slots
        for j in range(h):
            base = j * L
            for i in range(L):
                active_mask[base + i] = 1.0
        centred = backend.mul_plain(centred, active_mask)
    sq = backend.mul(centred, centred)
    var_bc = per_col_sum_then_broadcast(
        backend, sq, L=L, hidden_dim=h, num_slots=n_slots, scale=inv_h,
    )
    inv_sigma = backend.polyval(var_bc, absorbed)
    scaled = backend.mul(centred, inv_sigma)
    scaled = backend.mul_plain(scaled, gamma_vec)
    out = backend.add_plain(scaled, beta_vec)
    return out


# ---------------------------------------------------------------------------
# Phase 8k: multi-ciphertext col-major (for L=128 hidden=768 at N=2^16)
# ---------------------------------------------------------------------------
#
# Layout: hidden dim of size H is split into n_cts groups of cols_per_ct
# columns each, where cols_per_ct = num_slots // L.
# x_cts[k] holds slot[j*L + i] = X[i, k*cols_per_ct + j]  for j ∈ [0, cols_per_ct).
# The last ct may have fewer than cols_per_ct active columns; pad slots
# are kept zero.




def layernorm_colmajor_multi(
    backend: CKKSBackend,
    x_cts: Sequence[Ciphertext],
    *,
    L: int,
    hidden: int,
    invsqrt_power_coeffs: Sequence[float],
    invsqrt_interval: tuple,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> List[Ciphertext]:
    """LN over a multi-ct col-major row image."""
    from math import comb
    n_slots = backend.capabilities.n_slots
    cols_per_ct = _cols_per_ct(backend, L)
    n_cts = len(x_cts)
    h = hidden
    inv_h = 1.0 / h

    a, b = invsqrt_interval
    sscale = 2.0 / (b - a)
    sshift = -(a + b) / (b - a)
    n = len(invsqrt_power_coeffs)
    absorbed = [0.0] * n
    for k, c in enumerate(invsqrt_power_coeffs):
        for r in range(k + 1):
            absorbed[r] += float(c) * (sscale ** r) * (sshift ** (k - r)) * comb(k, r)

    # μ broadcast (cross-ct).
    mean_bc = per_col_sum_multi(
        backend, x_cts, L=L, hidden_per_ct=cols_per_ct, scale=inv_h,
    )
    centred = sub_multi(backend, x_cts, mean_bc)

    # Mask off pad cols in the final ct if hidden % cols_per_ct != 0.
    last_active = h - (n_cts - 1) * cols_per_ct
    if last_active < cols_per_ct:
        mask = [0.0] * n_slots
        for j in range(last_active):
            base = j * L
            for i in range(L):
                mask[base + i] = 1.0
        centred[-1] = backend.mul_plain(centred[-1], mask)

    sq = mul_multi(backend, centred, centred)
    var_bc = per_col_sum_multi(
        backend, sq, L=L, hidden_per_ct=cols_per_ct, scale=inv_h,
    )
    inv_sigma = backend.polyval(var_bc[0], absorbed)
    inv_sigma_list = [inv_sigma] * n_cts

    scaled = mul_multi(backend, centred, inv_sigma_list)

    out = []
    for k, sc in enumerate(scaled):
        lo = k * cols_per_ct
        hi = min(lo + cols_per_ct, h)
        chunk = hi - lo
        gv = [0.0] * n_slots
        bv = [0.0] * n_slots
        for j in range(chunk):
            base = j * L
            g = float(gamma[lo + j])
            b_ = float(beta[lo + j])
            for i in range(L):
                gv[base + i] = g
                bv[base + i] = b_
        s = backend.mul_plain(sc, gv)
        out.append(backend.add_plain(s, bv))
    return out


