"""HEonGPU CKKS backend.

Implements the minimal :class:`~fhe_thesis.encryption.backend.CKKSBackend`
surface needed by :mod:`fhe_thesis.encryption.ops_matrix` so the
matrix-packed Halevi-Shoup matmul can run end-to-end on a GPU via the
:mod:`fhe_thesis.encryption.heongpu_bindings` pybind11 wrapper.

Scope
-----
* Encrypt / decrypt of length-``num_slots`` plaintext vectors.
* Element-wise add / add-plain / mul-plain / mul-with-relinearise.
* Galois rotation by an arbitrary (integer) shift, decomposed into the
  power-of-two basis generated at startup.
* ``polyval``, ``matmul_plain``, ``dot``, ``sum_slots`` are intentionally
  **not implemented** in this first cut — they're only needed by the
  legacy token-packed code path. Calling them raises so we fail loudly
  if the protocol layer slips into the wrong path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import BackendCapabilities, CKKSBackend, Ciphertext


class HEonGPUBackend(CKKSBackend):
    """GPU CKKS backend backed by HEonGPU (Hopper-class GPUs).

    Parameters
    ----------
    poly_modulus_degree : int
        Ring dimension ``N``. Use 32768 for an LPAN BERT-Base layer
        (mid-depth chain) or 65536 if bootstrapping is required.
    q_prime_bits : sequence of int
        Bit-sizes of the *Q* primes (level chain). The first entry is
        traditionally the special 60-bit decryption prime. The number of
        entries minus the leading 60-bit prime gives the multiplicative
        depth available before bootstrapping.
    p_prime_bits : sequence of int, optional
        Bit-sizes of the *P* primes (key-switching modulus). Defaults to
        a single 60-bit prime, which is sufficient when no bootstrapping
        is configured.
    sec_none : bool, default False
        Pass ``True`` only for tiny demos (``N ≤ 4096``); HEonGPU rejects
        secure parameter sets at those sizes.
    scale_bits : int, optional
        log2 of the scale used when encoding plaintexts. Defaults to the
        bit-size of the second Q-prime (the one consumed by the first
        rescale), which keeps the scale stable across the chain.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 32768,
        q_prime_bits: Sequence[int] = (60,) + (40,) * 18,
        p_prime_bits: Sequence[int] = (60,),
        sec_none: bool = False,
        scale_bits: Optional[int] = None,
    ) -> None:
        # Imported lazily so the rest of the package still imports on a
        # box without the compiled wrapper.
        from . import heongpu_bindings as hg

        if poly_modulus_degree & (poly_modulus_degree - 1):
            raise ValueError(
                f"poly_modulus_degree must be a power of two, got {poly_modulus_degree}"
            )
        q_bits = list(q_prime_bits)
        p_bits = list(p_prime_bits)
        if len(q_bits) < 2:
            raise ValueError("need at least 2 Q primes (special + 1 level)")

        self._N = poly_modulus_degree
        self._num_slots = poly_modulus_degree // 2
        self._scale = float(2 ** (scale_bits if scale_bits is not None else q_bits[1]))

        self._ctx = hg.CKKSContext(self._N, q_bits, p_bits, sec_none=sec_none)
        kg = hg.KeyGenerator(self._ctx)
        self._sk = kg.generate_secret_key(self._ctx)
        self._pk = kg.generate_public_key(self._ctx, self._sk)
        self._rk = kg.generate_relin_key(self._ctx, self._sk)

        # Pre-generate Galois keys for ±1, ±2, ±4, … up to N/2 so any
        # arbitrary shift can be assembled by composition.
        max_log = (self._num_slots).bit_length() - 1
        pow2 = [1 << k for k in range(max_log)]
        shifts = pow2 + [-s for s in pow2]
        self._gk = kg.generate_galois_key(self._ctx, self._sk, shifts)

        self._encoder = hg.Encoder(self._ctx)
        self._encryptor = hg.Encryptor(self._ctx, self._pk)
        self._decryptor = hg.Decryptor(self._ctx, self._sk)
        self._ops = hg.Operator(self._ctx, self._encoder)
        self._hg = hg

        self.capabilities = BackendCapabilities(
            name="heongpu",
            supports_bootstrapping=True,
            supports_galois_rotations=True,
            n_slots=self._num_slots,
            initial_levels=len(q_bits) - 1,
        )

    # ── helpers ───────────────────────────────────────────────────────
    def _pad(self, values: Sequence[float]) -> List[float]:
        v = list(values)
        if len(v) > self._num_slots:
            raise ValueError(
                f"vector length {len(v)} exceeds num_slots {self._num_slots}"
            )
        if len(v) < self._num_slots:
            v.extend([0.0] * (self._num_slots - len(v)))
        return v

    def _encode(self, values: Sequence[float]):
        return self._encoder.encode(self._ctx, self._pad(values), self._scale)

    # ── encryption / decryption ───────────────────────────────────────
    def encrypt(self, values: Sequence[float]) -> Ciphertext:
        return self._encryptor.encrypt(self._ctx, self._encode(values))

    def decrypt(self, ct: Ciphertext) -> List[float]:
        pt = self._decryptor.decrypt(self._ctx, ct)
        # ``decode`` returns a Python list of floats already.
        return self._encoder.decode(pt)

    # ── linear ops ────────────────────────────────────────────────────
    # Note: HEonGPU operators are in-place. We clone the ciphertext
    # explicitly via re-encryption-of-zero-plus-add when an out-of-place
    # result is required, but in practice every call site of these ops
    # already treats the returned handle as the *result* (existing
    # OpenFHE backend has the same out-of-place convention).
    def add(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        out = self._clone(a)
        # Auto-match modulus levels so callers can mix freshly-multiplied
        # operands (one level lower) with un-rescaled ones.
        self._ops.add_inplace_match(out, b)
        return out

    def sub(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        out = self._clone(a)
        self._ops.sub_inplace(out, b)
        return out

    def add_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext:
        # HEonGPU has no add_plain primitive in the wrapper yet; encrypt
        # the constant and reuse via add_inplace_match (level-aware).
        out = self._clone(a)
        rhs = self.encrypt(plain)
        self._ops.add_inplace_match(out, rhs)
        return out

    def mul_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext:
        out = self._clone(a)
        self._ops.multiply_plain_inplace(out, self._encode(plain))
        # NOTE: scale doubled. The matrix-packed kernel accumulates several
        # mul_plain results before a single rescale at the call site. For
        # the cleanest semantics we rescale here, matching OpenFHE's
        # ``matmul_plain`` convention.
        self._ops.rescale_inplace(out)
        return out

    def mul(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        out = self._clone(a)
        self._ops.multiply_inplace(out, b)
        self._ops.relinearize_inplace(out, self._rk)
        self._ops.rescale_inplace(out)
        return out

    # ── rotation (composed power-of-two shifts, matches OpenFHE backend) ──
    def rotate(self, ct: Ciphertext, steps: int) -> Ciphertext:
        if steps == 0:
            return self._clone(ct)
        s = steps % self._num_slots
        if s > self._num_slots // 2:
            s -= self._num_slots
        sign = 1 if s > 0 else -1
        rem = abs(s)
        out = self._clone(ct)
        bit = 1
        while rem:
            if rem & 1:
                self._ops.rotate_rows_inplace(out, self._gk, sign * bit)
            rem >>= 1
            bit <<= 1
        return out

    # ── ops not yet ported (legacy path only) ─────────────────────────
    def polyval(self, ct: Ciphertext, power_coeffs: Sequence[float]) -> Ciphertext:
        raise NotImplementedError(
            "HEonGPUBackend.polyval not implemented yet — use the matrix-packed "
            "polynomial path or call this op via the OpenFHE backend."
        )

    def matmul_plain(
        self,
        ct: Ciphertext,
        weight: Sequence[Sequence[float]],
        bias: Optional[Sequence[float]] = None,
    ) -> Ciphertext:
        raise NotImplementedError(
            "HEonGPUBackend.matmul_plain (token-packed) not implemented; use "
            "fhe_thesis.encryption.ops_matrix.enc_linear_matrix on a "
            "MatrixPackedTensor instead."
        )

    def dot(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:  # pragma: no cover
        raise NotImplementedError("HEonGPUBackend.dot not implemented yet.")

    # ── ciphertext utilities ──────────────────────────────────────────
    def _clone(self, ct: Ciphertext) -> Ciphertext:
        """Return a fresh ciphertext with the same plaintext as ``ct``.

        HEonGPU operations are in-place; to honour the out-of-place
        contract of :class:`CKKSBackend` we materialise a working copy
        by adding a freshly-encrypted zero ciphertext. Keeps depth flat
        (no rescale).
        """
        zero = self._encryptor.encrypt(
            self._ctx,
            self._encoder.encode(
                self._ctx, [0.0] * self._num_slots, self._scale
            ),
        )
        self._ops.add_inplace(zero, ct)
        return zero

    # ── bootstrapping (Phase 4) ───────────────────────────────────────
    def configure_bootstrapping(
        self,
        CtoS_piece: int = 3,
        StoC_piece: int = 3,
        taylor_number: int = 11,
        less_key_mode: bool = True,
    ) -> None:
        self._ops.generate_bootstrapping_params(
            self._scale, CtoS_piece, StoC_piece, taylor_number, less_key_mode
        )
        # Replace galois key with one that covers bootstrap shifts.
        boot_shifts = self._ops.bootstrapping_key_indexs()
        kg = self._hg.KeyGenerator(self._ctx)
        self._gk = kg.generate_galois_key(self._ctx, self._sk, boot_shifts)

    def bootstrap(self, ct: Ciphertext) -> Ciphertext:
        return self._ops.regular_bootstrapping(ct, self._gk, self._rk)

    # ── introspection ─────────────────────────────────────────────────
    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def scale(self) -> float:
        return self._scale
