"""NEXUS-style ciphertext compression / decompression.

Port of NEXUS `MMEvaluator::enc_compress_ciphertext` and
``decompress_ciphertext`` (see cuda/src/matrix_mul.cu) using HEonGPU
primitives:
- ``encode_coeff``  (NEXUS-style polynomial-coefficient packing)
- ``apply_galois_elt`` (Galois automorphism with raw element)
- ``multiply_power_of_x_inplace`` (Phase 8a kernel)

# Algorithm

Compression:
    Pack N input scalars c_0..c_{N-1} as the coefficients of the plaintext
    polynomial p(x) = sum_j c_j x^j (via encode_coeff).
    Encrypt under CKKS: ct = (a, b) with b = a*s + p + e.
    Result: a single ciphertext storing N independent values in coefficients.

Decompression (the standard "subsum-expansion"):
    Returns N ciphertexts each containing one of the original c_j scalars
    broadcast across all slots. Uses log2(N) rounds of:
        ψ_t  : automorphism a(x) -> a(x^t),  t = N/2^i + 1
        x^k  : multiplication by x^k mod (x^N+1)
        + and split into two halves per round.

After round i we have 2^(i+1) ciphertexts; after the final round we have
N ciphertexts where the j-th one holds c_j replicated in every slot
(scaled by N because we started at coefficient scale, so a final divide
by N is required when interpreting the result).

The galois elements required are:
    galois_elts[i] = (N + 2^i) // 2^i   for i in [0, log2 N).
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def required_galois_elts(N: int) -> List[int]:
    """Galois elements needed for NEXUS decompress (one per round)."""
    log_n = int(math.log2(N))
    assert 1 << log_n == N, f"N must be power of 2, got {N}"
    return [(N + (1 << i)) // (1 << i) for i in range(log_n)]


def enc_compress(backend, values: Sequence[float], scale: float = None):
    """Encode ``values`` (length ≤ N) as polynomial coefficients and encrypt.

    Returns a single Ciphertext that decompresses to one constant-broadcast
    ciphertext per input value.
    """
    N = backend._N
    if scale is None:
        scale = backend._scale
    vals = list(values)
    if len(vals) < N:
        vals = vals + [0.0] * (N - len(vals))
    elif len(vals) > N:
        raise ValueError(f"Got {len(vals)} values, max is N={N}")
    pt = backend._encoder.encode_coeff(backend._ctx, vals, float(scale))
    ct = backend._encryptor.encrypt(backend._ctx, pt)
    return ct


def decompress(backend, ct, gk_decomp) -> List:
    """Decompress one packed ct into N broadcast ciphertexts.

    ``gk_decomp`` must be a GaloisKey containing the elements returned by
    :func:`required_galois_elts` for ``N = backend._N``.

    NOTE: the output ciphertexts are scaled by N relative to the input
    coefficient values. NEXUS handles this by setting
    ``res_col_ct.scale() *= N`` after the matmul; we follow the same
    convention upstream rather than mod-switching here.
    """
    N = backend._N
    log_n = int(math.log2(N))
    elts = required_galois_elts(N)
    ops = backend._ops

    # temp[a] is the a-th piece of the partial decomposition.
    temp = [ct]

    for i in range(log_n):
        galois_elt = elts[i]
        index_raw = (N << 1) - (1 << i)
        index = (index_raw * galois_elt) % (N << 1)
        new_temp = [None] * (len(temp) << 1)
        for a, t_a in enumerate(temp):
            # rotated = ψ_{galois_elt}(t_a)
            rotated = ops.apply_galois_elt(t_a, gk_decomp, galois_elt)
            # newtemp[a]            = t_a + rotated
            sum_ct = backend._clone(t_a)
            ops.add_inplace(sum_ct, rotated)
            new_temp[a] = sum_ct
            # shifted        = t_a       * x^{index_raw}
            shifted = backend._clone(t_a)
            ops.multiply_power_of_x_inplace(shifted, index_raw)
            # rotated_shifted = rotated  * x^{index}
            rs = backend._clone(rotated)
            ops.multiply_power_of_x_inplace(rs, index)
            # newtemp[a + len(temp)] = shifted + rotated_shifted
            ops.add_inplace(shifted, rs)
            new_temp[a + len(temp)] = shifted
        temp = new_temp

    return temp


# ---------------------------------------------------------------------------
# Phase 8d: plaintext-ciphertext MatMul on decompressed inputs
# ---------------------------------------------------------------------------


def matrix_mul(
    backend,
    W: np.ndarray,
    decompressed_x: List,
) -> List:
    """Compute y = W @ x homomorphically.

    Inputs
    ------
    W : (M, N) plaintext weight matrix.
    decompressed_x : list of N broadcast ciphertexts where ``decompressed_x[j]``
        is the decompression of input slot j (i.e. holds the polynomial
        ``p(x) = N * x_j`` whose slot interpretation is the constant ``N*x_j``
        broadcast across every slot).

    Returns
    -------
    list of M ciphertexts where ``y[i]`` broadcasts the scalar ``(W @ x)[i]``
    across every slot. The implicit ``N`` from decompression is cancelled here
    by pre-dividing the weights by ``N``.

    Cost: M * N ``mul_plain`` operations + M * (N-1) ``add_inplace``. Suitable
    for small/medium M, N. For large matmul use the column-block fold.
    """
    M, Ndim = W.shape
    if len(decompressed_x) != Ndim:
        raise ValueError(
            f"Expected {Ndim} decompressed ciphertexts, got {len(decompressed_x)}"
        )
    inv_N = 1.0 / float(backend._N)
    # Pre-scale weights to undo the implicit N from coefficient broadcast.
    W_scaled = W.astype(np.float64) * inv_N
    n_slots = backend._num_slots

    out_cts: List = []
    for i in range(M):
        # Accumulator: start with the j=0 term so we don't need an
        # encrypt-of-zero.
        w0 = float(W_scaled[i, 0])
        acc = backend.mul_plain(decompressed_x[0], [w0] * n_slots)
        for j in range(1, Ndim):
            wj = float(W_scaled[i, j])
            term = backend.mul_plain(decompressed_x[j], [wj] * n_slots)
            backend._ops.add_inplace_match(acc, term)
        out_cts.append(acc)
    return out_cts
