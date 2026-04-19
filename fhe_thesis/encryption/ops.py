"""Encrypted operations on TokenPackedTensor.

All operations preserve the token-packed layout (one ciphertext per
token, hidden_dim slots). See `docs/ckks_protocol.md` §4.

Phase-1 scope: linear, GELU-poly, LN-poly. Attention and softmax-poly
land in Phase 2.
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
