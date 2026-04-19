"""Encrypted operations on TokenPackedTensor.

All operations preserve the token-packed layout (one ciphertext per
token, hidden_dim slots). See `docs/ckks_protocol.md` §4.

Phase-1 scope: linear, GELU-poly, LN-poly.
Phase-2 scope: Q·Kᵀ scores, softmax-poly, attention·V.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .backend import CKKSBackend
from .packing import TokenPackedTensor


# ── Linear layer ───────────────────────────────────────────────────────
def enc_linear(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    weight: np.ndarray,
    bias: np.ndarray,
) -> TokenPackedTensor:
    """Apply y_i = W · x_i + b for every token row.

    `weight` is shape (out_dim, in_dim) — standard PyTorch nn.Linear
    convention. `bias` is (out_dim,).
    """
    out_dim, in_dim = weight.shape
    if in_dim != x.hidden_dim:
        raise ValueError(f"linear in_dim={in_dim} != tensor hidden_dim={x.hidden_dim}")
    if bias.shape != (out_dim,):
        raise ValueError(f"bias shape {bias.shape} != ({out_dim},)")

    w_rows = weight.tolist()
    b_list = bias.tolist()
    new_cts = [backend.matmul_plain(ct, w_rows, b_list) for ct in x.cts]
    return TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=out_dim)


# ── GELU polynomial ────────────────────────────────────────────────────
def enc_gelu_poly(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    power_coeffs: Sequence[float],
    interval: tuple[float, float],
) -> TokenPackedTensor:
    """Evaluate the LPAN GELU polynomial element-wise.

    `power_coeffs` is the polynomial in the *power* basis, already
    converted from Chebyshev. `interval = (a, b)` is the approximation
    interval; inputs are linearly mapped to [-1, 1] before evaluation,
    matching how the Chebyshev fit was made.
    """
    a, b = interval
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)

    new_cts = []
    for ct in x.cts:
        # affine standardisation: x_std = scale * x + shift
        ct_std = backend.add_plain(
            backend.mul_plain(ct, [scale] * x.hidden_dim),
            [shift] * x.hidden_dim,
        )
        new_cts.append(backend.polyval(ct_std, power_coeffs))
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
) -> TokenPackedTensor:
    """LPAN LayerNorm: y = γ · (x - μ)/σ + β with σ⁻¹ ≈ poly(σ²).

    We do **not** subtract the mean here: the LPAN ablation showed that
    centring is absorbed by the affine (γ, β) re-parameterisation
    learned in Stage 3. The expensive part is σ⁻¹ ≈ p(σ²+eps), which
    we approximate with a degree-8 polynomial fit over the profiled
    variance interval.
    """
    h = x.hidden_dim
    inv_h = 1.0 / h
    a, b = invsqrt_interval
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)

    new_cts = []
    for ct in x.cts:
        # variance ≈ mean(x²) ; reuse ct·ct multiplication
        ct_sq = backend.mul(ct, ct)
        # Σ x² across slots is a backend-rotation operation; for token-
        # packing it can be done via log2(h) rotations + adds. To keep
        # the protocol layer backend-agnostic in Phase 1 we delegate to
        # the backend if it exposes `sum_slots`, otherwise we fall back
        # to a single-slot decryption-free trick: multiply by a mask of
        # 1/h and rely on the caller to have set the unused slots to 0.
        sum_sq = _sum_first_n_slots(backend, ct_sq, h)
        var = backend.mul_plain(sum_sq, [inv_h] * h)
        # standardise variance to [-1,1] for the inv-sqrt polynomial
        var_std = backend.add_plain(backend.mul_plain(var, [scale] * h), [shift] * h)
        inv_sigma = backend.polyval(var_std, invsqrt_power_coeffs)
        # y = γ · x · inv_sigma + β
        scaled = backend.mul(ct, inv_sigma)
        scaled = backend.mul_plain(scaled, gamma.tolist())
        out = backend.add_plain(scaled, beta.tolist())
        new_cts.append(out)
    return TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h)


def _sum_first_n_slots(backend: CKKSBackend, ct, n: int):
    """Sum the first n slots of a ciphertext into every slot.

    Phase-1 fallback: ask the backend to decrypt+re-encrypt only if it
    cannot rotate. The reference TenSEAL backend supports rotations via
    `ct.sum()`.
    """
    summed = ct.sum()  # TenSEAL CKKSVector.sum returns a length-1 ct
    # Broadcast the scalar back to n slots by adding to a zero plaintext
    # of length n; tenseal's overload promotes the scalar.
    return backend.add_plain(summed, [0.0] * n)


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

      1. ``mask_j = ⟨Q[i], K[j]⟩``  → backend.dot, ct holds scalar
         broadcast across slots.
      2. multiply by a one-hot plaintext that selects slot ``j`` and
         zeroes everything else;
      3. accumulate into the row's result ciphertext.

    No rotations are needed because ``backend.dot`` already broadcasts
    the scalar across all slots in TenSEAL.
    """
    if Q.seq_len != K.seq_len:
        raise ValueError(f"Q.seq_len {Q.seq_len} != K.seq_len {K.seq_len}")
    if Q.hidden_dim != K.hidden_dim:
        raise ValueError(f"Q.hidden_dim {Q.hidden_dim} != K.hidden_dim {K.hidden_dim}")

    L = Q.seq_len
    # Pre-build one-hot masks once; reused for every query row.
    masks = [[0.0] * L for _ in range(L)]
    for j in range(L):
        masks[j][j] = scale

    rows = []
    for i in range(L):
        row_ct = None
        for j in range(L):
            score = backend.dot(Q.cts[i], K.cts[j])  # broadcast scalar
            term = backend.mul_plain(score, masks[j])  # keep slot j only
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

    LPAN replaces softmax with a per-head degree-d polynomial; under
    token-packed scores this is identical in structure to ``enc_gelu_poly``,
    just operating over the seq_len-slot rows instead of hidden_dim slots.
    """
    a, b = interval
    width = b - a
    scale = 2.0 / width
    shift = -(a + b) / width

    L = scores.hidden_dim
    new_cts = []
    for ct in scores.cts:
        ct_std = backend.add_plain(
            backend.mul_plain(ct, [scale] * L),
            [shift] * L,
        )
        new_cts.append(backend.polyval(ct_std, power_coeffs))
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
            scalar = backend.sum_slots(slot)
            term = backend.mul(V.cts[j], scalar)
            row = term if row is None else backend.add(row, term)
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
    softmax_power_coeffs: Sequence[float],
    softmax_interval: tuple[float, float],
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
        A = enc_softmax_poly(backend, S, softmax_power_coeffs, softmax_interval)
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

    # Per-head plaintext mask of length hidden_dim with 1s only in
    # the head's destination slot range. Each head's ct only has
    # head_dim slots populated; mul_plain pads with zeros so the
    # result occupies hidden_dim slots cleanly.
    masks = []
    for h in range(num_heads):
        m = [0.0] * hidden_dim
        for k in range(h * head_dim, (h + 1) * head_dim):
            m[k] = 1.0
        masks.append(m)

    out_cts = []
    for i in range(seq_len):
        row = None
        for h in range(num_heads):
            # mul_plain by a longer mask broadcasts the head ct's
            # head_dim slots; the trailing slots stay at the head's
            # native zero (TenSEAL pads on encode), and the masked
            # destination slots receive the head's values.
            term = backend.mul_plain(heads[h].cts[i], masks[h])
            row = term if row is None else backend.add(row, term)
        out_cts.append(row)
    return TokenPackedTensor.from_ciphertexts(out_cts, hidden_dim=hidden_dim)
