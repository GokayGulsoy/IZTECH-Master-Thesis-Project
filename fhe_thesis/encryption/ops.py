"""Encrypted operations on TokenPackedTensor.

All operations preserve the token-packed layout (one ciphertext per
token, hidden_dim slots). See `docs/ckks_protocol.md` §4.

Phase-1 scope: linear, GELU-poly, LN-poly.
Phase-2 scope: Q·Kᵀ scores, softmax-poly, attention·V.

Threading (O5)
--------------
Token-level loops in ``enc_linear``, ``enc_gelu_poly``, and ``enc_ln_poly``
are embarrassingly parallel — each ciphertext is independent. These functions
accept an optional ``n_jobs`` argument (default 1). Set ``n_jobs=-1`` to use
all CPUs, or pass an explicit count. Attention ops remain serial because of
cross-token Q·Kᵀ dependencies.
"""

from __future__ import annotations

from math import comb
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence
import os

import numpy as np

from .backend import CKKSBackend
from .coefficients import PolyCoeffs
from .packing import TokenPackedTensor


# ── Affine substitution helper ──────────────────────────────────────────
def _resolve_jobs(n_jobs: Optional[int]) -> int:
    """Convert n_jobs sentinel to an actual worker count."""
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return os.cpu_count() or 1
    return max(1, n_jobs)


def _absorb_affine(
    power_coeffs: Sequence[float], scale: float, shift: float
) -> list[float]:
    """Return ``c'`` such that ``p'(x) = p(scale·x + shift)``.

    Used to fold the standardisation map ``x → (scale·x + shift)`` into
    the polynomial coefficients themselves, saving one multiplicative
    CKKS level per polynomial op.

    For ``p(y) = Σ_i c_i y^i`` the substitution ``y = scale·x + shift``
    expands to ``c'_k = Σ_{i≥k} c_i · C(i,k) · scale^k · shift^{i-k}``.
    """
    c = list(power_coeffs)
    d = len(c) - 1
    new = [0.0] * (d + 1)
    for k in range(d + 1):
        acc = 0.0
        sk = scale**k
        for i in range(k, d + 1):
            acc += c[i] * comb(i, k) * sk * (shift ** (i - k))
        new[k] = acc
    return new


# ── Linear layer ───────────────────────────────────────────────────────
def enc_linear(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    weight: np.ndarray,
    bias: np.ndarray,
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """Apply y_i = W · x_i + b for every token row.

    `weight` is shape (out_dim, in_dim) — standard PyTorch nn.Linear
    convention. `bias` is (out_dim,).

    `n_jobs` controls parallelism across token rows (O5).
    """
    out_dim, in_dim = weight.shape
    if in_dim != x.hidden_dim:
        raise ValueError(f"linear in_dim={in_dim} != tensor hidden_dim={x.hidden_dim}")
    if bias.shape != (out_dim,):
        raise ValueError(f"bias shape {bias.shape} != ({out_dim},)")

    w_rows = weight.tolist()
    b_list = bias.tolist()

    workers = _resolve_jobs(n_jobs)
    if workers == 1:
        new_cts = [backend.matmul_plain(ct, w_rows, b_list) for ct in x.cts]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            new_cts = list(pool.map(lambda ct: backend.matmul_plain(ct, w_rows, b_list), x.cts))
    return TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=out_dim)


# ── GELU polynomial ────────────────────────────────────────────────────
def enc_gelu_poly(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    power_coeffs: Sequence[float],
    interval: tuple[float, float],
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """Evaluate the LPAN GELU polynomial element-wise.

    `power_coeffs` is the polynomial in the *power* basis, already
    converted from Chebyshev. `interval = (a, b)` is the approximation
    interval; the standardisation ``x → (2x − a − b)/(b − a) ∈ [-1,1]``
    is *folded into the coefficients* via :func:`_absorb_affine` so it
    consumes zero CKKS multiplicative levels.

    `n_jobs` controls parallelism across token rows (O5).
    """
    a, b = interval
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    absorbed = _absorb_affine(power_coeffs, scale, shift)

    workers = _resolve_jobs(n_jobs)
    if workers == 1:
        new_cts = [backend.polyval(ct, absorbed) for ct in x.cts]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            new_cts = list(pool.map(lambda ct: backend.polyval(ct, absorbed), x.cts))
    return TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=x.hidden_dim)


# ── LayerNorm polynomial ───────────────────────────────────────────────
def enc_ln_poly(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    invsqrt_power_coeffs: Sequence[float],
    invsqrt_interval: tuple[float, float],
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """LPAN LayerNorm: y = γ · (x - μ)/σ + β with σ⁻¹ ≈ poly(σ²+eps).

    Mirrors :class:`fhe_thesis.models.activations.PolynomialLayerNorm`
    exactly: subtract the per-token mean, compute variance from the
    centred tensor, then apply the inv-sqrt polynomial. The
    standardisation map ``var → (2·var − a − b)/(b − a)`` is folded
    into the polynomial coefficients via :func:`_absorb_affine` so it
    consumes zero CKKS levels.
    """
    h = x.hidden_dim
    inv_h = 1.0 / h
    a, b = invsqrt_interval
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    absorbed_invsqrt = _absorb_affine(invsqrt_power_coeffs, scale, shift)

    gamma_list = gamma.tolist()
    beta_list = beta.tolist()

    def _process_token(ct):
        mean_bc = _sum_first_n_slots(backend, ct, h, scale=inv_h)
        centred = backend.sub(ct, mean_bc)
        centred_sq = backend.mul(centred, centred)
        var = _sum_first_n_slots(backend, centred_sq, h, scale=inv_h)
        inv_sigma = backend.polyval(var, absorbed_invsqrt)
        scaled = backend.mul(centred, inv_sigma)
        scaled = backend.mul_plain(scaled, gamma_list)
        return backend.add_plain(scaled, beta_list)

    workers = _resolve_jobs(n_jobs)
    if workers == 1:
        new_cts = [_process_token(ct) for ct in x.cts]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            new_cts = list(pool.map(_process_token, x.cts))
    return TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h)


def _sum_first_n_slots(backend: CKKSBackend, ct, n: int, scale: float = 1.0):
    """Sum the first *n* slots of a ciphertext, scale, broadcast to *n* slots.

    Uses the backend's ``sum_slots`` to produce a scalar (broadcast across
    slots in OpenFHE; size-1 ct in TenSEAL), then a ``matmul_plain`` with
    an (n, 1) column vector of ``scale`` to broadcast to n slots. Cost:
    1 multiplicative level (already accounted for in ``DEPTH_COST['ln_poly']``).
    """
    summed = backend.sum_slots(ct)
    # weight is (out=n, in=1), each row [scale] → output[j] = scale * summed[0]
    return backend.broadcast_first_slot(summed, n, scale)


# ── Attention primitives (Phase 2) ─────────────────────────────────────
def enc_qk_scores(
    backend: CKKSBackend,
    Q: TokenPackedTensor,
    K: TokenPackedTensor,
    scale: float,
) -> TokenPackedTensor:
    """Compute scaled scores S[i,j] = scale · ⟨Q[i], K[j]⟩.

    Output is a token-packed tensor of shape (seq_len, seq_len): one
    ciphertext per query row, with seq_len slots holding the scores
    for that row against every key.

    Strategy
    --------
    For each query row ``Q[i]``, we compute ``seq_len`` inner products
    against ``K[0], K[1], …, K[L-1]`` and pack them slot-wise into a
    single result ciphertext. Each inner product is materialised by:

      1. ``s_j = ⟨Q[i], K[j]⟩``  → ``backend.dot``, returns a size-1 ct.
      2. broadcast the scalar into slot ``j`` via plaintext matmul with
         an ``(L, 1)`` weight matrix that is ``scale`` in row ``j`` and
         ``0`` elsewhere — yields a size-*L* ct with the scaled score
         in slot ``j`` and zeros elsewhere.
      3. accumulate (free additions) into the row's result ciphertext.

    Cost: 1 ct·ct multiplication (dot) + 1 plaintext matmul = 2
    multiplicative levels per row, matching ``DEPTH_COST['qk_scores']``.
    """
    if Q.seq_len != K.seq_len:
        raise ValueError(f"Q.seq_len {Q.seq_len} != K.seq_len {K.seq_len}")
    if Q.hidden_dim != K.hidden_dim:
        raise ValueError(f"Q.hidden_dim {Q.hidden_dim} != K.hidden_dim {K.hidden_dim}")

    L = Q.seq_len

    rows = []
    for i in range(L):
        row_ct = None
        for j in range(L):
            score = backend.dot(Q.cts[i], K.cts[j])  # slot 0 holds the scalar
            # Place scaled score at slot j of an L-vector (zeros elsewhere).
            term = backend.place_scaled_at_slot(score, j, L, scale)
            row_ct = term if row_ct is None else backend.add(row_ct, term)
        rows.append(row_ct)
    return TokenPackedTensor.from_ciphertexts(rows, hidden_dim=L)


def enc_softmax_poly(
    backend: CKKSBackend,
    scores: TokenPackedTensor,
    power_coeffs: Sequence[float],
    interval: tuple[float, float],
) -> TokenPackedTensor:
    """Apply the LPAN softmax polynomial element-wise on each row.

    Uses :func:`_absorb_affine` so the interval standardisation
    consumes zero multiplicative levels.
    """
    a, b = interval
    width = b - a
    scale = 2.0 / width
    shift = -(a + b) / width
    absorbed = _absorb_affine(power_coeffs, scale, shift)

    L = scores.hidden_dim
    new_cts = [backend.polyval(ct, absorbed) for ct in scores.cts]
    return TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=L)


def enc_attention_apply(
    backend: CKKSBackend,
    attn: TokenPackedTensor,
    V: TokenPackedTensor,
) -> TokenPackedTensor:
    """Compute output[i] = Σ_j attn[i, j] · V[j].

    Parameters
    ----------
    attn : TokenPackedTensor
        Shape (seq_len, seq_len) — one ct per query row, seq_len slots.
    V : TokenPackedTensor
        Shape (seq_len, head_dim) — one ct per token row, head_dim slots.

    Strategy
    --------
    For each (i, j) we extract the scalar ``attn[i, j]`` by masking +
    summing slots, multiply it into ``V[j]`` (a head_dim-vector ct),
    and accumulate. Cost per output row: ``seq_len × (mul_plain + sum
    + ct·ct + add)`` operations.
    """
    if attn.seq_len != V.seq_len:
        raise ValueError(f"attn.seq_len {attn.seq_len} != V.seq_len {V.seq_len}")
    if attn.hidden_dim != attn.seq_len:
        raise ValueError(
            "attn must be square (seq_len, seq_len); "
            f"got ({attn.seq_len}, {attn.hidden_dim})"
        )

    L = attn.seq_len
    head_dim = V.hidden_dim
    # Pre-build one-hot masks for slot extraction.
    masks = [[0.0] * L for _ in range(L)]
    for j in range(L):
        masks[j][j] = 1.0

    out_cts = []
    for i in range(L):
        row = None
        attn_i = attn.cts[i]
        for j in range(L):
            # Extract scalar attn[i, j]: keep slot j, sum to broadcast.
            slot = backend.mul_plain(attn_i, masks[j])
            scalar = backend.sum_slots(slot)  # slot 0 holds the scalar
            # Broadcast to head_dim slots so it can multiply V[j].
            scalar_broadcast = backend.broadcast_first_slot(scalar, head_dim, 1.0)
            # Lazy relin (O6): defer relinearisation until after accumulation.
            if hasattr(backend, "mul_no_relin"):
                term = backend.mul_no_relin(V.cts[j], scalar_broadcast)
            else:
                term = backend.mul(V.cts[j], scalar_broadcast)
            row = term if row is None else backend.add(row, term)
        # Relinearise once after the full accumulation loop (O6).
        if hasattr(backend, "relinearize") and row is not None:
            row = backend.relinearize(row)
        out_cts.append(row)
    return TokenPackedTensor.from_ciphertexts(out_cts, hidden_dim=head_dim)


def enc_self_attention(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    Wq: np.ndarray,
    bq: np.ndarray,
    Wk: np.ndarray,
    bk: np.ndarray,
    Wv: np.ndarray,
    bv: np.ndarray,
    Wo: np.ndarray,
    bo: np.ndarray,
    softmax_coeffs: PolyCoeffs,
    num_heads: int,
) -> TokenPackedTensor:
    """End-to-end encrypted multi-head self-attention.

    Splits the **plaintext** Q/K/V weight matrices row-wise into
    ``num_heads`` slices of ``(head_dim, hidden)`` and runs each head
    independently as its own ``enc_linear → enc_qk_scores →
    enc_softmax_poly → enc_attention_apply`` pipeline. The output of
    each head is a token-packed tensor of width ``head_dim``; we
    concatenate them under FHE by zero-padding the slot positions and
    summing — no decryption ever happens server-side, preserving the
    PF-SR (Pure-FHE Single-Round) protocol guarantee.

    The scale ``1/√head_dim`` is folded into ``enc_qk_scores`` so the
    softmax polynomial sees scores in its profiled interval.
    """
    if Wq.shape != Wk.shape or Wq.shape != Wv.shape:
        raise ValueError("Q/K/V weight shapes must match")
    hidden = Wq.shape[0]
    if hidden % num_heads != 0:
        raise ValueError(f"hidden {hidden} not divisible by num_heads {num_heads}")
    head_dim = hidden // num_heads
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)

    head_outputs = []  # list of TokenPackedTensor[head_dim] per head
    for h in range(num_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        Wq_h, bq_h = Wq[s:e, :], bq[s:e]
        Wk_h, bk_h = Wk[s:e, :], bk[s:e]
        Wv_h, bv_h = Wv[s:e, :], bv[s:e]

        Q_h = enc_linear(backend, x, Wq_h, bq_h)
        K_h = enc_linear(backend, x, Wk_h, bk_h)
        V_h = enc_linear(backend, x, Wv_h, bv_h)

        S = enc_qk_scores(backend, Q_h, K_h, scale=inv_sqrt_d)

        # Select per-head softmax coefficients if available
        if softmax_coeffs.per_head:
            head_power_coeffs = softmax_coeffs.power_coeffs[h].tolist()
        else:
            head_power_coeffs = softmax_coeffs.power_coeffs.tolist()

        A = enc_softmax_poly(
            backend, S, head_power_coeffs, softmax_coeffs.interval
        )
        H = enc_attention_apply(backend, A, V_h)
        head_outputs.append(H)

    concat = _concat_heads_zero_pad(backend, head_outputs, hidden)
    return enc_linear(backend, concat, Wo, bo)


def _concat_heads_zero_pad(
    backend: CKKSBackend,
    heads: list[TokenPackedTensor],
    hidden_dim: int,
) -> TokenPackedTensor:
    """Concatenate per-head outputs token-wise into one ciphertext per
    token of width ``hidden_dim``, **without** ever decrypting.

    For each head ``h`` and each token ``i`` we multiply the head's
    ciphertext by a one-hot-block plaintext of length ``hidden_dim``
    that is 1 inside the head's slot range and 0 elsewhere. Adding the
    masked head ciphertexts yields the concatenation in the right
    slots. Cost: ``num_heads × seq_len × (mul_plain + add)`` ops.
    """
    num_heads = len(heads)
    head_dim = heads[0].hidden_dim
    seq_len = heads[0].seq_len
    if head_dim * num_heads != hidden_dim:
        raise ValueError(
            f"head_dim×num_heads ({head_dim}×{num_heads}) != hidden_dim ({hidden_dim})"
        )

    # Per-head zero-pad weight matrices. ``W_h`` has shape
    # ``(hidden_dim, head_dim)``; row ``k`` selects the head's input
    # slot ``k - h·head_dim`` if ``k`` falls in the head's destination
    # range, else zero. Multiplying the head's size-``head_dim`` ct by
    # this plaintext matmul produces a size-``hidden_dim`` ct with the
    # head's values placed in slots ``[h·head_dim, (h+1)·head_dim)``
    # and zeros elsewhere — a strict zero-pad, no decryption.
    weights = []
    for h in range(num_heads):
        W = [[0.0] * head_dim for _ in range(hidden_dim)]
        for j in range(head_dim):
            W[h * head_dim + j][j] = 1.0
        weights.append(W)

    out_cts = []
    for i in range(seq_len):
        row = None
        for h in range(num_heads):
            term = backend.matmul_plain(heads[h].cts[i], weights[h])
            row = term if row is None else backend.add(row, term)
        out_cts.append(row)
    return TokenPackedTensor.from_ciphertexts(out_cts, hidden_dim=hidden_dim)
