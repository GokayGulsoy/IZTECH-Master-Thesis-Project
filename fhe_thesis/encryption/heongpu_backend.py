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

import hashlib
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple

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
        bootstrap_hamming_weight: Optional[int] = None,
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
        # Maximum depth a ciphertext can reach before bootstrap (or rescale
        # below the chain). Each rescale drops one Q prime.
        self._max_depth = len(q_bits) - 1

        self._ctx = hg.CKKSContext(self._N, q_bits, p_bits, sec_none=sec_none)
        kg = hg.KeyGenerator(self._ctx)
        if bootstrap_hamming_weight is not None:
            self._sk = kg.generate_secret_key_h(self._ctx, bootstrap_hamming_weight)
        else:
            self._sk = kg.generate_secret_key(self._ctx)
        self._pk = kg.generate_public_key(self._ctx, self._sk)
        self._rk = kg.generate_relin_key(self._ctx, self._sk)

        # Pre-generate Galois keys for ±1, ±2, ±4, … up to N/2 so any
        # arbitrary shift can be assembled by composition.
        max_log = (self._num_slots).bit_length() - 1
        pow2 = [1 << k for k in range(max_log)]
        shifts = pow2 + [-s for s in pow2]
        self._gk = kg.generate_galois_key(self._ctx, self._sk, shifts)
        self._registered_shifts = set(shifts)

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

        # Cache for Halevi-Shoup diagonals — keyed by sha1(weight.bytes()).
        self._diag_cache: Dict[str, Tuple[List[Optional[List[float]]], int]] = {}
        self._diag_cache_lock = Lock()
        # Becomes True after configure_bootstrapping() succeeds; gates
        # auto-refresh inside mul/mul_plain.
        self._bootstrap_ready = False

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
        da = self._ops.depth(out)
        db = self._ops.depth(b)
        if da < db:
            for _ in range(db - da):
                self._ops.mod_drop_inplace_ct(out)
            rhs = b
        elif db < da:
            rhs = self._clone(b)
            for _ in range(da - db):
                self._ops.mod_drop_inplace_ct(rhs)
        else:
            rhs = b
        self._ops.sub_inplace(out, rhs)
        return out

    def add_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext:
        # HEonGPU has no add_plain primitive in the wrapper yet; encrypt
        # the constant and reuse via add_inplace_match (level-aware).
        out = self._clone(a)
        rhs = self.encrypt(plain)
        self._ops.add_inplace_match(out, rhs)
        return out

    def mul_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext:
        # Auto-refresh if the next rescale would push us off the chain.
        if self._bootstrap_ready and self._ops.depth(a) >= self._max_depth:
            a = self.bootstrap(a)
        out = self._clone(a)
        pt = self._encode(plain)
        # Plaintext must live at the same modulus level as the ciphertext;
        # encoded plaintexts start at level 0 so drop until depths match.
        target_depth = self._ops.depth(out)
        while self._ops.depth_of_plaintext(pt) < target_depth:
            self._ops.mod_drop_inplace_pt(pt)
        self._ops.multiply_plain_inplace(out, pt)
        # NOTE: scale doubled. The matrix-packed kernel accumulates several
        # mul_plain results before a single rescale at the call site. For
        # the cleanest semantics we rescale here, matching OpenFHE's
        # ``matmul_plain`` convention.
        self._ops.rescale_inplace(out)
        return out

    def mul(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        # Auto-refresh either operand if a single mul+rescale would
        # exhaust the chain.
        if self._bootstrap_ready:
            if self._ops.depth(a) >= self._max_depth:
                a = self.bootstrap(a)
            if self._ops.depth(b) >= self._max_depth:
                b = self.bootstrap(b)
        out = self._clone(a)
        da = self._ops.depth(out)
        db = self._ops.depth(b)
        if da < db:
            for _ in range(db - da):
                self._ops.mod_drop_inplace_ct(out)
            rhs = b
        elif db < da:
            rhs = self._clone(b)
            for _ in range(da - db):
                self._ops.mod_drop_inplace_ct(rhs)
        else:
            rhs = b
        self._ops.multiply_inplace(out, rhs)
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
        # Fast path: if a Galois key for this exact shift exists, do
        # one rotation. Otherwise fall back to the bit-decomposition.
        registered = getattr(self, "_registered_shifts", None)
        if registered is not None and s in registered:
            out = self._clone(ct)
            self._ops.rotate_rows_inplace(out, self._gk, s)
            return out
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

    # ── high-level ops ────────────────────────────────────────────────
    def polyval(self, ct: Ciphertext, power_coeffs: Sequence[float]) -> Ciphertext:
        """Evaluate p(x) = Σ c_i · x^i with depth ≈ ⌈log₂(deg)⌉ + 1.

        Strategy: precompute x^(2^i) by binary doubling, then build each
        x^k by binary-tree multiplication of the powers indicated by the
        bits of k. Finally sum c_k · x^k. Total multiplicative depth is
        ⌈log₂(deg)⌉ + 1 — same order as Paterson-Stockmeyer.
        """
        coeffs = list(power_coeffs)
        D = len(coeffs) - 1
        if D < 0:
            return self.mul_plain(ct, [0.0] * self._num_slots)
        if D == 0:
            return self.add_plain(
                self.mul_plain(ct, [0.0] * self._num_slots),
                [coeffs[0]] * self._num_slots,
            )

        # x^(2^i) for i in [0, L]
        L = (D).bit_length() - 1
        pow2: Dict[int, Ciphertext] = {1: ct}
        for i in range(1, L + 1):
            prev = pow2[1 << (i - 1)]
            pow2[1 << i] = self.mul(prev, prev)

        # Build x^k for each k in [1, D] used in the polynomial.
        x_pow: Dict[int, Ciphertext] = {1: ct}
        for k in range(2, D + 1):
            if coeffs[k] == 0.0:
                continue
            bits = [1 << b for b in range(k.bit_length()) if (k >> b) & 1]
            cur: List[Ciphertext] = [pow2[b] for b in bits]
            while len(cur) > 1:
                nxt: List[Ciphertext] = []
                for j in range(0, len(cur), 2):
                    if j + 1 < len(cur):
                        nxt.append(self.mul(cur[j], cur[j + 1]))
                    else:
                        nxt.append(cur[j])
                cur = nxt
            x_pow[k] = cur[0]

        result: Optional[Ciphertext] = None
        for k in range(1, D + 1):
            if coeffs[k] == 0.0:
                continue
            term = self.mul_plain(x_pow[k], [coeffs[k]] * self._num_slots)
            result = term if result is None else self.add(result, term)
        if result is None:
            result = self.mul_plain(ct, [0.0] * self._num_slots)
        return self.add_plain(result, [coeffs[0]] * self._num_slots)

    def matmul_plain(
        self,
        ct: Ciphertext,
        weight: Sequence[Sequence[float]],
        bias: Optional[Sequence[float]] = None,
    ) -> Ciphertext:
        """Halevi-Shoup diagonal matmul with cached diagonals.

        Mirrors :meth:`OpenFHEBackend.matmul_plain` so the existing
        token-packed ops work unchanged on the GPU backend.
        """
        out_dim = len(weight)
        if out_dim == 0:
            raise ValueError("weight is empty")
        in_dim = len(weight[0])

        target = max(out_dim, in_dim)
        n = 1
        while n < target:
            n <<= 1
        if n > self._num_slots:
            raise ValueError(
                f"matmul dim n={n} exceeds num_slots={self._num_slots}"
            )

        w_arr = np.asarray(weight, dtype=np.float64)
        w_hash = hashlib.sha1(w_arr.tobytes()).hexdigest()
        with self._diag_cache_lock:
            cached = self._diag_cache.get(w_hash)
        if cached is None:
            diagonals: List[Optional[List[float]]] = []
            for i in range(n):
                diag = [0.0] * self._num_slots
                any_nz = False
                for k in range(out_dim):
                    col = (k + i) % n
                    if col < in_dim:
                        v = float(w_arr[k, col])
                        diag[k] = v
                        if v != 0.0:
                            any_nz = True
                diagonals.append(diag if any_nz else None)
            with self._diag_cache_lock:
                self._diag_cache[w_hash] = (diagonals, n)
        else:
            diagonals, n = cached

        # Cyclic-replicate input so the period-n pattern fills every slot.
        # (CKKS rotation is mod num_slots, not mod n, so we need real
        # periodicity across the entire ring.)
        x = ct
        cur = in_dim
        while cur < self._num_slots:
            x = self.add(x, self.rotate(x, -cur))
            cur <<= 1

        result: Optional[Ciphertext] = None
        for i, diag in enumerate(diagonals):
            if diag is None:
                continue
            rot_x = x if i == 0 else self.rotate(x, i)
            term = self.mul_plain(rot_x, diag)
            result = term if result is None else self.add(result, term)
        if result is None:
            result = self.mul_plain(ct, [0.0] * self._num_slots)
        if bias is not None:
            b = list(bias) + [0.0] * (self._num_slots - len(bias))
            result = self.add_plain(result, b)
        return result

    # ── NEXUS-style coefficient packing ──────────────────────────────
    def encrypt_coeff(self, coeffs: Sequence[float]) -> Ciphertext:
        """Encrypt a vector as the *coefficients* of the plaintext polynomial.

        ``coeffs`` is padded with zeros to length N. Pair with
        :meth:`coeff_matvec` for NEXUS-style coefficient-packed matmul.
        """
        v = list(coeffs)
        if len(v) > self._N:
            raise ValueError(
                f"coeffs length {len(v)} exceeds poly degree N={self._N}"
            )
        if len(v) < self._N:
            v.extend([0.0] * (self._N - len(v)))
        pt = self._encoder.encode_coeff(self._ctx, v, self._scale)
        return self._encryptor.encrypt(self._ctx, pt)

    def decrypt_coeff(self, ct: Ciphertext) -> List[float]:
        """Decrypt and return the polynomial coefficients (length N)."""
        pt = self._decryptor.decrypt(self._ctx, ct)
        return self._encoder.decode_coeff(pt)

    def _encode_coeff_pad(self, coeffs: Sequence[float]):
        v = list(coeffs)
        if len(v) > self._N:
            raise ValueError(
                f"coeffs length {len(v)} exceeds poly degree N={self._N}"
            )
        if len(v) < self._N:
            v.extend([0.0] * (self._N - len(v)))
        return self._encoder.encode_coeff(self._ctx, v, self._scale)

    def coeff_matvec(
        self,
        ct_x_coeff: Ciphertext,
        weight,
        *,
        in_dim: int,
        rescale: bool = True,
    ) -> Ciphertext:
        """NEXUS-style coefficient-packed matrix–vector product.

        Computes ``y = W · x`` for ``W ∈ R^{m × n}``, ``x ∈ R^n`` with a
        **single** ciphertext × plaintext multiplication. The trick:

        - ``ct_x_coeff`` is from :meth:`encrypt_coeff` with coefficients
          ``[x_0, x_1, …, x_{n-1}, 0, 0, …]`` (n = ``in_dim``).
        - The plaintext polynomial encodes W with row i reversed at
          coefficient indices ``[i·n, i·n + n − 1]``::

              pt_W[i·n + (n − 1 − j)] = W[i, j]

        - Then ``coef[i·n + n − 1]`` of ``ct_x · pt_W (mod X^N + 1)``
          equals ``Σ_j x_j · W[i, j] = ⟨W_i, x⟩``.

        Constraint: ``m · n ≤ N``. Otherwise split W and call multiple
        times (caller's responsibility for Phase 2; Phase 3 wraps this).
        """
        import numpy as np
        W = np.asarray(weight, dtype=np.float64)
        if W.ndim != 2:
            raise ValueError(f"weight must be 2-D, got {W.shape}")
        m, n = W.shape
        if n != in_dim:
            raise ValueError(f"W in_dim={n} != in_dim arg {in_dim}")
        if m * n > self._N:
            raise ValueError(
                f"m·n = {m * n} exceeds poly degree N={self._N}; "
                "split W and run multiple multiplications"
            )

        # Build the W-poly: row i reversed at offset i·n.
        w_poly = [0.0] * self._N
        for i in range(m):
            base = i * n
            for j in range(n):
                w_poly[base + (n - 1 - j)] = float(W[i, j])
        pt_w = self._encode_coeff_pad(w_poly)

        out = self._ops.clone_ct(ct_x_coeff)
        target_depth = self._ops.depth(out)
        while self._ops.depth_of_plaintext(pt_w) < target_depth:
            self._ops.mod_drop_inplace_pt(pt_w)
        self._ops.multiply_plain_inplace(out, pt_w)
        if rescale:
            self._ops.rescale_inplace(out)
        return out

    def decrypt_coeff_extract(
        self, ct: Ciphertext, in_dim: int, out_dim: int
    ) -> List[float]:
        """Decrypt + extract m output values from a :meth:`coeff_matvec` result.

        Reads coefficients ``[n−1, 2n−1, …, m·n − 1]``.
        """
        coeffs = self.decrypt_coeff(ct)
        return [coeffs[(i + 1) * in_dim - 1] for i in range(out_dim)]

    def coeff_to_slot(self, ct: Ciphertext) -> List[Ciphertext]:
        """Homomorphic conversion: coefficient-encoded → 2× slot-encoded.

        Returns a list of 2 ciphertexts:

        - ``out[0]``: slot vector ``[0, N/2)`` holds polynomial coefficients ``[0, N/2)``
        - ``out[1]``: slot vector ``[0, N/2)`` holds polynomial coefficients ``[N/2, N)``

        Useful for chaining a :meth:`coeff_matvec` output into slot-domain
        polynomial / element-wise ops.

        Requires :meth:`configure_bootstrapping` to have been called.
        Input ``ct`` must be at depth 0 (top of the chain) — that is
        where the precomputed CtoS plaintext matrices live.
        """
        if not self._bootstrap_ready:
            raise RuntimeError(
                "coeff_to_slot requires configure_bootstrapping() first "
                "(populates the CtoS BSGS matrices and rotation keys)"
            )
        if self._ops.depth(ct) != 0:
            raise RuntimeError(
                f"coeff_to_slot requires depth-0 input, got depth={self._ops.depth(ct)}; "
                f"call coeff_matvec(..., rescale=False) and skip the rescale, or use "
                f"coeff_matvec_to_slot()"
            )
        return self._ops.coeff_to_slot(ct, self._gk)

    def coeff_matvec_to_slot(
        self,
        ct_x_coeff: Ciphertext,
        weight,
        *,
        in_dim: int,
    ) -> List[Ciphertext]:
        """NEXUS-style fused matvec + slot conversion.

        Computes ``y = W · x`` via :meth:`coeff_matvec` (no trailing
        rescale, so the result stays at depth 0), then runs
        :meth:`coeff_to_slot` so the inner products land in the slot
        domain at indices ``[i·n + n − 1 for i in range(m)]``.

        Returns the 2-ct list from CtoS. Use ``out[0]`` if ``m·n ≤ N/2``.
        """
        ct_y = self.coeff_matvec(ct_x_coeff, weight, in_dim=in_dim, rescale=False)
        # CtoS plaintext matrices live at depth 0 (top of chain). Our
        # multiply-without-rescale leaves ct at depth 0 with a doubled
        # scale; clear the rescale_required_ flag so the rotations
        # inside CtoS don't reject it.
        self._ops.clear_rescale_required(ct_y)
        cts = self.coeff_to_slot(ct_y)
        # CtoS internally rescales twice but the doubled input scale
        # propagates: outputs sit at scale²= 2^(2·log2(scale)). One more
        # rescale brings them back to the canonical scale so downstream
        # mul_plain plaintexts (encoded at the canonical scale) align.
        for c in cts:
            self._ops.set_rescale_required(c)
            self._ops.rescale_inplace(c)
        return cts

    @staticmethod
    def _bitrev(i: int, bits: int) -> int:
        r = 0
        for b in range(bits):
            r = (r << 1) | ((i >> b) & 1)
        return r

    def nexus_target_slots(self, in_dim: int, out_dim: int) -> List[int]:
        """Slot indices where coeff_matvec_to_slot lands (W·x)[0..out_dim).

        Useful for building gather masks. All values land in ``out[0]``
        of :meth:`coeff_matvec_to_slot` provided ``in_dim·out_dim ≤ N/2``.
        """
        log_n = int(self._num_slots).bit_length() - 1
        return [self._bitrev((i + 1) * in_dim - 1, log_n) for i in range(out_dim)]

    def _bsgs_decompose(
        self, shifts: Sequence[int], baby_size: Optional[int] = None
    ) -> Tuple[int, List[int], List[int]]:
        """Pick BSGS baby step ``B`` and return (B, baby_set, giant_set).

        ``baby_set`` = unique baby-step rotations used (in [0, B)).
        ``giant_set`` = unique giant-step rotations used (multiples of B).
        """
        N = self._num_slots
        uniq = sorted({int(s) % N for s in shifts})
        if baby_size is None:
            import math
            baby_size = max(1, int(math.isqrt(max(1, len(uniq)))))
        babies = set()
        giants = set()
        for s in uniq:
            b = s % baby_size
            g = s - b
            babies.add(b)
            giants.add(g % N)
        return baby_size, sorted(babies), sorted(giants)

    def gather_slots(
        self,
        ct: Ciphertext,
        src_indices: Sequence[int],
        *,
        dst_indices: Optional[Sequence[int]] = None,
        baby_size: Optional[int] = None,
    ) -> Ciphertext:
        """Permute slots: ``out[dst_indices[i]] = ct[src_indices[i]]``.

        If ``dst_indices`` is ``None``, defaults to ``[0, 1, ..., m-1]``.

        Hoisted BSGS implementation. With ``U`` distinct shifts and baby
        step ``B = ⌈√U⌉``, this needs ``|baby_set| + |giant_set| ≈ 2√U``
        rotations and ``U`` mul_plains, instead of ``U`` rotations. This is
        the difference between gathering 256 distinct shifts at ~32 keys
        vs 256 keys (avoids OOM on the Galois eval keys).

        Multiplicative depth: 1.
        """
        N = self._num_slots
        srcs = [int(s) % N for s in src_indices]
        m = len(srcs)
        if dst_indices is None:
            dsts = list(range(m))
        else:
            dsts = [int(d) % N for d in dst_indices]
            if len(dsts) != m:
                raise ValueError(
                    f"dst_indices length {len(dsts)} != src_indices length {m}"
                )

        # We want result[dst] = ct[src]. With our convention
        # rotate(x, k)[i] = x[(i+k) mod N], we need a total shift of
        # k = (src - dst) mod N. Decompose k = g + b, g multiple of B,
        # b ∈ [0, B). Pre-rotate ct by each b, mask at position (dst+g),
        # then giant-rotate by g.
        # mask_{g,b}[(dst + g) mod N] = 1 contributes ct[(dst+g+b) mod N]
        # = ct[src] when dst+g+b ≡ src (mod N).
        all_k = [(srcs[i] - dsts[i]) % N for i in range(m)]
        B, _, _ = self._bsgs_decompose(all_k, baby_size=baby_size)

        # Bucket: giant g → list of (b, mask_position)
        giant_buckets: Dict[int, Dict[int, List[int]]] = {}
        babies_used = set()
        for src, dst, k in zip(srcs, dsts, all_k):
            b = k % B
            g = (k - b) % N
            babies_used.add(b)
            giant_buckets.setdefault(g, {}).setdefault(b, []).append((dst + g) % N)

        # Pre-rotate ct by each baby step.
        baby_rots: Dict[int, Ciphertext] = {}
        for b in babies_used:
            baby_rots[b] = self.rotate(ct, b) if b else ct

        # Register Galois keys: babies (positive) ∪ giants (positive)
        needed_keys = set(babies_used) | set(giant_buckets.keys())
        # rotate() expects signed shifts; convert (we use positive only here)
        self.register_rotation_keys(needed_keys)

        # Build masked sums per giant, then giant-rotate, accumulate.
        result: Optional[Ciphertext] = None
        for g, b_to_positions in giant_buckets.items():
            acc: Optional[Ciphertext] = None
            for b, positions in b_to_positions.items():
                mask = [0.0] * N
                for p in positions:
                    mask[p] = 1.0
                term = self.mul_plain(baby_rots[b], mask)
                acc = term if acc is None else self.add(acc, term)
            assert acc is not None
            moved = self.rotate(acc, g) if g else acc
            result = moved if result is None else self.add(result, moved)

        if result is None:
            return self.mul_plain(ct, [0.0] * N)
        return result

    def nexus_linear(
        self,
        ct_x_coeff: Ciphertext,
        weight,
        *,
        in_dim: int,
        bias: Optional[Sequence[float]] = None,
        register_keys: bool = True,
    ) -> "Ciphertext":
        """Full NEXUS linear pipeline: coefficient input → slot output."""
        import numpy as np
        W = np.asarray(weight, dtype=np.float64)
        out_dim = W.shape[0]
        targets = self.nexus_target_slots(in_dim, out_dim)
        cts = self.coeff_matvec_to_slot(ct_x_coeff, W, in_dim=in_dim)
        # gather_slots will call register_rotation_keys with the small
        # baby+giant set (~2√U keys) instead of U keys.
        gathered = self.gather_slots(cts[0], targets)
        if bias is not None:
            b_pad = list(bias) + [0.0] * (self._num_slots - len(bias))
            gathered = self.add_plain(gathered, b_pad)
        return gathered

    def slot_to_coeff(self, ct0: Ciphertext, ct1: Ciphertext) -> Ciphertext:
        """Homomorphic conversion: 2× slot-encoded → coefficient-encoded.

        Inverse of :meth:`coeff_to_slot`; takes the 2 halves and produces
        a single coefficient-encoded ciphertext of length N.
        """
        if not self._bootstrap_ready:
            raise RuntimeError(
                "slot_to_coeff requires configure_bootstrapping() first"
            )
        return self._ops.slot_to_coeff(ct0, ct1, self._gk)

    def dot(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        """Inner product ⟨a, b⟩ broadcast across slots."""
        return self.sum_slots(self.mul(a, b))

    def sum_slots(self, ct: Ciphertext) -> Ciphertext:
        """Sum every slot, broadcast across all slots (log-N rotate+add)."""
        out = self._clone(ct)
        step = 1
        while step < self._num_slots:
            shifted = self.rotate(out, step)
            out = self.add(out, shifted)
            step <<= 1
        return out

    def broadcast_first_slot(
        self, ct: Ciphertext, n: int, scale: float = 1.0
    ) -> Ciphertext:
        """Mask first n slots of an already-broadcast scalar (after sum_slots).

        sum_slots leaves every slot holding the same value; we just keep
        the first n and zero the rest, scaled by ``scale``. One mul_plain
        instead of the n×mul_plain default — saves levels and time.
        """
        mask = [scale] * n + [0.0] * (self._num_slots - n)
        return self.mul_plain(ct, mask)

    def place_scaled_at_slot(
        self, ct: Ciphertext, slot: int, n: int, scale: float = 1.0
    ) -> Ciphertext:
        """Mask out slot 0 of `ct` (broadcast scalar) into a one-hot at `slot`.

        Caller is expected to pass a ct whose slot 0 holds the scalar (e.g.
        after `dot`/`sum_slots`). One mul_plain.
        """
        mask = [0.0] * self._num_slots
        if slot < n:
            mask[slot] = scale
        return self.mul_plain(ct, mask)

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
        self._ops.add_inplace_match(zero, ct)
        return zero

    # ── bootstrapping (Phase 4) ───────────────────────────────────────
    def configure_bootstrapping(
        self,
        CtoS_piece: int = 3,
        StoC_piece: int = 3,
        taylor_number: int = 11,
        less_key_mode: bool = True,
        extra_shifts: Optional[Sequence[int]] = None,
    ) -> None:
        self._ops.generate_bootstrapping_params(
            self._scale, CtoS_piece, StoC_piece, taylor_number, less_key_mode
        )
        # Merge boot shifts with the existing ±2^k power-of-two set so
        # matmul / rotate paths keep working after configure_bootstrapping.
        boot_shifts = list(self._ops.bootstrapping_key_indexs())
        max_log = (self._num_slots).bit_length() - 1
        pow2 = [1 << k for k in range(max_log)]
        extras = list(extra_shifts) if extra_shifts is not None else []
        all_shifts = sorted(set(boot_shifts + pow2 + [-s for s in pow2] + extras))
        kg = self._hg.KeyGenerator(self._ctx)
        self._gk = kg.generate_galois_key(self._ctx, self._sk, all_shifts)
        self._bootstrap_ready = True
        self._registered_shifts = set(all_shifts)

    def register_rotation_keys(self, shifts: Sequence[int]) -> int:
        """Add rotation keys for ``shifts`` (in addition to ±2^k and bootstrap keys).

        Returns the number of *newly* added shifts. Re-runs
        ``generate_galois_key`` on the union, which is a few seconds.
        Used by NEXUS to cache one-shot rotation keys for the bit-rev
        gather pattern produced by :meth:`coeff_matvec_to_slot`.
        """
        existing = getattr(self, "_registered_shifts", None)
        if existing is None:
            max_log = (self._num_slots).bit_length() - 1
            pow2 = [1 << k for k in range(max_log)]
            existing = set(pow2 + [-s for s in pow2])
            if getattr(self, "_bootstrap_ready", False):
                existing |= set(self._ops.bootstrapping_key_indexs())
        wanted = set(int(s) % self._num_slots for s in shifts)
        # Normalise to signed (HEonGPU expects shift in (-N/2, N/2]).
        signed = {s if s <= self._num_slots // 2 else s - self._num_slots for s in wanted}
        new = signed - existing
        if not new:
            return 0
        merged = sorted(existing | signed)
        # Free the old Galois key BEFORE allocating the new one — each
        # key holds ~|merged| × 32MB at N=2^16, L=31, so a careless
        # double-allocation can blow the 75GB pool.
        self._gk = None
        import gc
        gc.collect()
        kg = self._hg.KeyGenerator(self._ctx)
        self._gk = kg.generate_galois_key(self._ctx, self._sk, merged)
        self._registered_shifts = set(merged)
        return len(new)

    def bootstrap(self, ct: Ciphertext) -> Ciphertext:
        """Refresh a CKKS ciphertext to a low-depth state.

        HEonGPU's `regular_bootstrapping` requires the input to be at the
        very bottom of the modulus chain. We mod-drop as needed so the
        caller doesn't have to keep track of depth themselves.
        """
        out = self._clone(ct)
        while self._ops.depth(out) < self._max_depth:
            self._ops.mod_drop_inplace_ct(out)
        return self._ops.regular_bootstrapping(out, self._gk, self._rk)

    # ── introspection ─────────────────────────────────────────────────
    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def scale(self) -> float:
        return self._scale
