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
    # NOTE: do NOT pre-scale W by 1/N. The decompressed ct holds polynomial
    # ``p(x) = N*x_j`` (only coeff 0 nonzero). After mul_plain with constant
    # ``w`` and summing over j, the output ct has ``coeff[0] = N * (W @ x)[i]``
    # with all other coeffs zero. Slot-domain interpretation = mean over the N
    # polynomial coefficients = ``(W @ x)[i]`` directly. ``inv_N`` is kept here
    # only so callers can pre-scale weights themselves if they prefer to read
    # ``coeff[0]`` instead of the slot mean.
    del inv_N
    Nring = backend._N
    ops = backend._ops
    enc = backend._encoder
    ctx = backend._ctx
    scale = backend._scale

    def _encode_const_coeff(w: float):
        """Encode a constant polynomial p(x) = w as a coeff-domain plaintext.

        ``encode_coeff`` puts each input value into a successive polynomial
        coefficient, so we just supply ``[w, 0, 0, ..., 0]``.
        """
        coeffs = [float(w)] + [0.0] * (Nring - 1)
        return enc.encode_coeff(ctx, coeffs, scale)

    out_cts: List = []
    for i in range(M):
        # j = 0 term — establishes the accumulator at the post-rescale depth.
        pt0 = _encode_const_coeff(W[i, 0])
        acc = backend._mul_plain_pt(decompressed_x[0], pt0)
        for j in range(1, Ndim):
            pt = _encode_const_coeff(W[i, j])
            term = backend._mul_plain_pt(decompressed_x[j], pt)
            ops.add_inplace_match(acc, term)
        out_cts.append(acc)
    return out_cts


# ---------------------------------------------------------------------------
# Phase 8d-2: optimized linear via coeff-domain inner product
# ---------------------------------------------------------------------------


def _encode_weight_row_inner_product(backend, w: np.ndarray):
    """Encode a length-N weight vector ``w`` so that ``(x · enc(w))[0] = <x, w>``.

    The constant coefficient of the negacyclic polynomial product
    ``(X · W) mod (X^N + 1)`` equals::

        (x · w)[0] = x_0 w_0  +  Σ_{k=1}^{N-1} (-1) · x_k · w_{N-k}

    so by setting ``W[0] = w_0`` and ``W[k] = -w_{N-k}`` for k ≥ 1, the
    constant coefficient becomes ``Σ_j x_j w_j`` — the desired inner product.
    """
    N = backend._N
    if w.shape[0] != N:
        raise ValueError(f"weight vector length {w.shape[0]} != N={N}")
    enc_coeffs = np.empty(N, dtype=np.float64)
    enc_coeffs[0] = float(w[0])
    enc_coeffs[1:] = -w[N - 1 : 0 : -1]  # k=1..N-1: -w_{N-k}
    return backend._encoder.encode_coeff(
        backend._ctx, enc_coeffs.tolist(), backend._scale
    )


def linear_compressed(
    backend,
    W: np.ndarray,
    x_compressed,
) -> List:
    """Optimized linear ``y = W @ x`` taking the COMPRESSED input ciphertext.

    Inputs
    ------
    W : (M, N) plaintext weight matrix.
    x_compressed : ONE Ciphertext holding ``Σ x_j X^j`` (output of
        :func:`enc_compress` with ``len(values) = N``).

    Returns
    -------
    list of M ciphertexts where ``y[i]`` has the scalar ``(W @ x)[i]`` at
    polynomial coefficient 0 (and zeros at the other coefficients up to
    encryption noise). Equivalent slot-domain interpretation:
    ``decoded.mean() / N == (W @ x)[i] / N``  — i.e. the ciphertext encodes
    the *constant polynomial* whose value is ``(W @ x)[i]``, scaled by 1/N
    in the slot domain because the constant polynomial has only its 0th
    Fourier coefficient nonzero. Callers reading slot-mean get the value
    directly (sum over N coeffs / N = const poly value).

    Cost: M ``mul_plain`` operations (one per output row). No decompress
    needed — this is the NEXUS linear-layer trick.
    """
    M, Ndim = W.shape
    if Ndim != backend._N:
        raise ValueError(f"W width {Ndim} must equal ring N={backend._N}")
    out: List = []
    for i in range(M):
        pt_w = _encode_weight_row_inner_product(backend, W[i])
        out.append(backend._mul_plain_pt(x_compressed, pt_w))
    return out


# ---------------------------------------------------------------------------
# Phase 8e: fold per-scalar output cts back into a single packed ct
# ---------------------------------------------------------------------------


def fold_outputs_to_packed(
    backend,
    out_cts: List,
    *,
    start_index: int = 0,
) -> "object":
    """Combine M ciphertexts (each with a scalar at coeff[0]) into one ct.

    Each input ct ``out_cts[i]`` holds a polynomial whose coefficient 0 is
    the scalar ``y_i`` (other coefficients ~0, as produced by
    :func:`linear_compressed`). The result is a single ciphertext holding
    ``Σ_i y_i X^{start_index + i}`` — i.e. the packed-polynomial layout that
    :func:`enc_compress` would produce if we encrypted the plaintext vector
    ``[0, ..., 0, y_0, y_1, ..., y_{M-1}, 0, ...]``.

    Suitable input for the next :func:`linear_compressed` call.

    Cost: M-1 ``multiply_power_of_x`` ops + M-1 ``add_inplace``.
    """
    if not out_cts:
        raise ValueError("out_cts is empty")
    ops = backend._ops
    M = len(out_cts)

    # i = 0 term: shift by start_index (skip the multiply if start_index == 0).
    acc = backend._clone(out_cts[0])
    if start_index != 0:
        ops.multiply_power_of_x_inplace(acc, start_index)

    for i in range(1, M):
        shifted = backend._clone(out_cts[i])
        ops.multiply_power_of_x_inplace(shifted, start_index + i)
        ops.add_inplace_match(acc, shifted)
    return acc
