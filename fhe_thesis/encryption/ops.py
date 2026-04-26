"""Encrypted operations on TokenPackedTensor.

All operations preserve the token-packed layout (one ciphertext per
token, hidden_dim slots). See `docs/ckks_protocol.md` §4.

Phase-1 scope: linear, GELU-poly, LN-poly.
Phase-2 scope: Q·Kᵀ scores, softmax-poly, attention·V.

Threading (O5)
--------------
Token-level loops in ``enc_linear``, ``enc_gelu_poly``, ``enc_ln_poly``,
and all attention ops are embarrassingly parallel — each ciphertext or
attention head is independent. All functions accept an optional ``n_jobs``
argument (default 1). Set ``n_jobs=-1`` to use all CPUs, or pass an
explicit count. ``enc_self_attention`` distributes heads across threads
and subdivides remaining threads for inner row-level parallelism.
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
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """Compute scaled scores S[i,j] = scale · ⟨Q[i], K[j]⟩.

    Output is a token-packed tensor of shape (seq_len, seq_len): one
    ciphertext per query row, with seq_len slots holding the scores
    for that row against every key.

    ``n_jobs`` parallelises the outer loop over query rows (O5).
    """
    if Q.seq_len != K.seq_len:
        raise ValueError(f"Q.seq_len {Q.seq_len} != K.seq_len {K.seq_len}")
    if Q.hidden_dim != K.hidden_dim:
        raise ValueError(f"Q.hidden_dim {Q.hidden_dim} != K.hidden_dim {K.hidden_dim}")

    L = Q.seq_len

    def _compute_row(i):
        row_ct = None
        for j in range(L):
            score = backend.dot(Q.cts[i], K.cts[j])
            term = backend.place_scaled_at_slot(score, j, L, scale)
            row_ct = term if row_ct is None else backend.add(row_ct, term)
        return row_ct

    workers = _resolve_jobs(n_jobs)
    if workers == 1:
        rows = [_compute_row(i) for i in range(L)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_compute_row, range(L)))
    return TokenPackedTensor.from_ciphertexts(rows, hidden_dim=L)


def enc_softmax_poly(
    backend: CKKSBackend,
    scores: TokenPackedTensor,
    power_coeffs: Sequence[float],
    interval: tuple[float, float],
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """Apply the LPAN softmax polynomial element-wise on each row.

    Uses :func:`_absorb_affine` so the interval standardisation
    consumes zero multiplicative levels.

    ``n_jobs`` parallelises across score rows (O5).
    """
    a, b = interval
    width = b - a
    scale = 2.0 / width
    shift = -(a + b) / width
    absorbed = _absorb_affine(power_coeffs, scale, shift)

    L = scores.hidden_dim
    workers = _resolve_jobs(n_jobs)
    if workers == 1:
        new_cts = [backend.polyval(ct, absorbed) for ct in scores.cts]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            new_cts = list(pool.map(lambda ct: backend.polyval(ct, absorbed), scores.cts))
    return TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=L)


def enc_attention_apply(
    backend: CKKSBackend,
    attn: TokenPackedTensor,
    V: TokenPackedTensor,
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """Compute output[i] = Σ_j attn[i, j] · V[j].

    Parameters
    ----------
    attn : TokenPackedTensor
        Shape (seq_len, seq_len) — one ct per query row, seq_len slots.
    V : TokenPackedTensor
        Shape (seq_len, head_dim) — one ct per token row, head_dim slots.
    n_jobs : int
        Row-level parallelism (O5).
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
    masks = [[0.0] * L for _ in range(L)]
    for j in range(L):
        masks[j][j] = 1.0

    def _compute_row(i):
        row = None
        attn_i = attn.cts[i]
        for j in range(L):
            slot = backend.mul_plain(attn_i, masks[j])
            scalar = backend.sum_slots(slot)
            scalar_broadcast = backend.broadcast_first_slot(scalar, head_dim, 1.0)
            if hasattr(backend, "mul_no_relin"):
                term = backend.mul_no_relin(V.cts[j], scalar_broadcast)
            else:
                term = backend.mul(V.cts[j], scalar_broadcast)
            row = term if row is None else backend.add(row, term)
        if hasattr(backend, "relinearize") and row is not None:
            row = backend.relinearize(row)
        return row

    workers = _resolve_jobs(n_jobs)
    if workers == 1:
        out_cts = [_compute_row(i) for i in range(L)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out_cts = list(pool.map(_compute_row, range(L)))
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
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """End-to-end encrypted multi-head self-attention.

    Splits the **plaintext** Q/K/V weight matrices row-wise into
    ``num_heads`` slices and runs each head independently. Heads are
    processed **in parallel** when ``n_jobs > 1`` (O5), with remaining
    threads subdivided for inner row-level parallelism.

    The scale ``1/√head_dim`` is folded into ``enc_qk_scores``.
    """
    if Wq.shape != Wk.shape or Wq.shape != Wv.shape:
        raise ValueError("Q/K/V weight shapes must match")
    hidden = Wq.shape[0]
    if hidden % num_heads != 0:
        raise ValueError(f"hidden {hidden} not divisible by num_heads {num_heads}")
    head_dim = hidden // num_heads
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)

    workers = _resolve_jobs(n_jobs)
    # Distribute threads: heads in parallel, remaining for inner ops.
    head_workers = min(workers, num_heads)
    inner_jobs = max(1, workers // head_workers) if head_workers > 1 else workers

    def _process_head(h):
        s, e = h * head_dim, (h + 1) * head_dim
        Wq_h, bq_h = Wq[s:e, :], bq[s:e]
        Wk_h, bk_h = Wk[s:e, :], bk[s:e]
        Wv_h, bv_h = Wv[s:e, :], bv[s:e]

        Q_h = enc_linear(backend, x, Wq_h, bq_h, n_jobs=inner_jobs)
        K_h = enc_linear(backend, x, Wk_h, bk_h, n_jobs=inner_jobs)
        V_h = enc_linear(backend, x, Wv_h, bv_h, n_jobs=inner_jobs)

        S = enc_qk_scores(backend, Q_h, K_h, scale=inv_sqrt_d, n_jobs=inner_jobs)

        if softmax_coeffs.per_head:
            head_power_coeffs = softmax_coeffs.power_coeffs[h].tolist()
        else:
            head_power_coeffs = softmax_coeffs.power_coeffs.tolist()

        A = enc_softmax_poly(
            backend, S, head_power_coeffs, softmax_coeffs.interval, n_jobs=inner_jobs
        )
        return enc_attention_apply(backend, A, V_h, n_jobs=inner_jobs)

    if head_workers > 1:
        with ThreadPoolExecutor(max_workers=head_workers) as pool:
            head_outputs = list(pool.map(_process_head, range(num_heads)))
    else:
        head_outputs = [_process_head(h) for h in range(num_heads)]

    concat = _concat_heads_zero_pad(backend, head_outputs, hidden, n_jobs=workers)
    return enc_linear(backend, concat, Wo, bo, n_jobs=workers)


def _concat_heads_zero_pad(
    backend: CKKSBackend,
    heads: list[TokenPackedTensor],
    hidden_dim: int,
    n_jobs: Optional[int] = 1,
) -> TokenPackedTensor:
    """Concatenate per-head outputs token-wise into one ciphertext per
    token of width ``hidden_dim``, **without** ever decrypting.

    When the backend supports ``rotate()``, each head's ciphertext is
    slot-shifted to its destination range and added — costing only
    ``num_heads`` rotations + additions per token (zero multiplicative
    levels). Falls back to ``matmul_plain`` when rotation is unavailable.
    """
    num_heads = len(heads)
    head_dim = heads[0].hidden_dim
    seq_len = heads[0].seq_len
    if head_dim * num_heads != hidden_dim:
        raise ValueError(
            f"head_dim×num_heads ({head_dim}×{num_heads}) != hidden_dim ({hidden_dim})"
        )

    use_rotation = backend.capabilities.supports_galois_rotations

    if use_rotation:
        # Rotation-based concat: shift each head to its slot range
        # then add. Cost: num_heads rotations per token, 0 mult levels.
        def _concat_token(i):
            row = heads[0].cts[i]  # head 0 is already at slots 0..head_dim-1
            for h in range(1, num_heads):
                # Rotate right by h*head_dim → places head's slots at
                # [h*head_dim, (h+1)*head_dim).
                shifted = backend.rotate(heads[h].cts[i], -(h * head_dim))
                row = backend.add(row, shifted)
            return row
    else:
        # Fallback: matmul_plain approach (costs 1 multiplicative level).
        weights = []
        for h in range(num_heads):
            W = [[0.0] * head_dim for _ in range(hidden_dim)]
            for j in range(head_dim):
                W[h * head_dim + j][j] = 1.0
            weights.append(W)

        def _concat_token(i):
            row = None
            for h in range(num_heads):
                term = backend.matmul_plain(heads[h].cts[i], weights[h])
                row = term if row is None else backend.add(row, term)
            return row

    workers = _resolve_jobs(n_jobs)
    if workers == 1:
        out_cts = [_concat_token(i) for i in range(seq_len)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out_cts = list(pool.map(_concat_token, range(seq_len)))
    return TokenPackedTensor.from_ciphertexts(out_cts, hidden_dim=hidden_dim)
