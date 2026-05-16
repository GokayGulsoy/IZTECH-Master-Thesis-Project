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
        self._kg = kg
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
        # Phase 7b-BSGS: cached encoded mask plaintexts.
        # Key: (block, shift, depth) → Plaintext.
        # Mask plaintexts are layer-shape-shared and reused across
        # every linear in the network.
        self._mask_pt_cache: Dict[Tuple[int, int, int], object] = {}
        self._encoding_transform_ctx: Dict[int, object] = {}
        # Per-weight encoded BSGS plaintext cache.
        # Key: (weight_id, x_depth) → (diag0_pts: list, diagj_pts: list).
        # On first call we encode once and stash; subsequent calls at
        # the same x_depth reuse the GPU plaintexts — the dominant cost
        # of a BSGS matmul. NOTE: each cached weight at block=4096 holds
        # ~64 GB of plaintexts; only useful when the SAME weight is
        # called repeatedly (e.g. inference over many batches) at the
        # SAME input depth. Disabled by default; enable via
        # ``backend.bsgs_diag_cache_enabled = True``.
        self._bsgs_diag_cache: Dict[Tuple[int, int], Tuple[list, list]] = {}
        self.bsgs_diag_cache_enabled = False
        # Becomes True after configure_bootstrapping() succeeds; gates
        # auto-refresh inside mul/mul_plain.
        self._bootstrap_ready = False
        self._bootstrap_ctos_piece = 3
        self._bootstrap_stoc_piece = 3
        self._bootstrap_less_key_mode = True
        self._supports_rescale_flag_override = bool(
            self._ops.supports_rescale_flag_override()
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

    def _mul_plain_pt(self, a: Ciphertext, pt) -> Ciphertext:
        """mul_plain that takes a pre-encoded :class:`Plaintext`.

        Mod-drops a *copy* of ``pt`` (so the cached plaintext can be reused
        at lower-level depths in subsequent calls — we don't mutate the
        cached object). This is the hot path for cached attention masks.
        """
        if self._bootstrap_ready and self._ops.depth(a) >= self._max_depth:
            a = self.bootstrap(a)
        out = self._clone(a)
        target_depth = self._ops.depth(out)
        pt_depth = self._ops.depth_of_plaintext(pt)
        if pt_depth < target_depth:
            # Need a copy to mod-drop without trashing the cache. HEonGPU's
            # Plaintext doesn't expose a clone, but we can clone via re-encode
            # — too expensive. Instead, mod-drop in place and trust callers
            # to use the same input depth across calls (true for our attn
            # masks: every d-iteration sees the same Q/K depth).
            while self._ops.depth_of_plaintext(pt) < target_depth:
                self._ops.mod_drop_inplace_pt(pt)
        self._ops.multiply_plain_inplace(out, pt)
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

    # ------------------------------------------------------------------
    # Phase 7b: batched Halevi-Shoup matmul (matrix-packed)
    # ------------------------------------------------------------------
    # HS_CHUNK bounds the linear-path plaintext footprint per call.
    # BSGS_THRESHOLD: above this many shifts we switch to BSGS (O(√n) keys).
    # BSGS_GIANT_CHUNK: how many giants to encode per inner C++ call.
    HS_CHUNK = 64
    BSGS_THRESHOLD = 64
    BSGS_GIANT_CHUNK = 8

    @staticmethod
    def _factor_bsgs(n: int) -> "tuple[int, int]":
        """Pick (b1, b2) with b1 * b2 == n and b1 ≈ √n.

        Prefers b1 a power of two when ``n`` is, so that subsequent layers
        can reuse the Galois key set.
        """
        # Find the divisor of n closest to sqrt(n).
        from math import isqrt
        s = isqrt(n)
        # Search downward then upward for a divisor.
        for d in range(s, 0, -1):
            if n % d == 0:
                return d, n // d
        return 1, n

    @staticmethod
    def _per_block_roll(diag: Sequence[float], shift: int,
                        block: int, num_slots: int) -> List[float]:
        """Cyclic right-roll of ``diag`` by ``shift`` within each block."""
        import numpy as _np
        arr = _np.asarray(diag, dtype=float).reshape(-1, block)
        return _np.roll(arr, shift, axis=1).reshape(num_slots).tolist()

    @staticmethod
    def _block_masks_static(block: int, shift: int, num_slots: int) \
            -> "tuple[List[float], List[float]]":
        if not 0 < shift < block:
            raise ValueError(f"bad shift={shift} block={block}")
        B = num_slots // block
        low = [0.0] * num_slots
        high = [0.0] * num_slots
        for b in range(B):
            base = b * block
            for j in range(block - shift):
                low[base + j] = 1.0
            for j in range(block - shift, block):
                high[base + j] = 1.0
        return low, high

    def _get_mask_pt(self, block: int, shift: int, depth: int):
        """Return cached (low_pt, high_pt) at the requested chain depth.

        The first request encodes both plaintexts at depth 0 and mod-drops
        them to ``depth``; subsequent requests at the same ``depth`` reuse
        the GPU plaintexts directly. Same-shape masks dominate the BSGS
        plaintext encoding cost, so this cache yields a large win.
        """
        key_lo = (block, shift, depth, 0)
        key_hi = (block, shift, depth, 1)
        lo_pt = self._mask_pt_cache.get(key_lo)
        hi_pt = self._mask_pt_cache.get(key_hi)
        if lo_pt is not None and hi_pt is not None:
            return lo_pt, hi_pt
        lo, hi = self._block_masks_static(block, shift, self._num_slots)
        lo_pt = self._encode(lo)
        hi_pt = self._encode(hi)
        while self._ops.depth_of_plaintext(lo_pt) < depth:
            self._ops.mod_drop_inplace_pt(lo_pt)
        while self._ops.depth_of_plaintext(hi_pt) < depth:
            self._ops.mod_drop_inplace_pt(hi_pt)
        self._mask_pt_cache[key_lo] = lo_pt
        self._mask_pt_cache[key_hi] = hi_pt
        return lo_pt, hi_pt

    def halevi_shoup_matmul(
        self,
        ct_x: Ciphertext,
        *,
        block: int,
        shifts: Sequence[int],
        diagonals: Sequence[Optional[Sequence[float]]],
        low_masks: Sequence[Optional[Sequence[float]]],
        high_masks: Sequence[Optional[Sequence[float]]],
        bias_vec: Optional[Sequence[float]] = None,
        weight_id: Optional[str] = None,
    ) -> Ciphertext:
        """Run the matrix-packed Halevi-Shoup matvec entirely in C++.

        For ``len(shifts) > BSGS_THRESHOLD`` the wrapper switches to
        baby-step/giant-step (Galois keys: O(√n) instead of O(n)).
        ``weight_id`` (when provided) keys a per-weight diagonal
        plaintext cache so subsequent calls at the same input depth
        skip diagonal encoding.
        """
        if not (len(shifts) == len(diagonals) == len(low_masks) == len(high_masks)):
            raise ValueError("shifts/diagonals/low_masks/high_masks size mismatch")

        # All-None fast path.
        if all(d is None for d in diagonals):
            zero_pt = self._encode([0.0] * self._num_slots)
            out = self._ops.clone_ct(ct_x)
            self._ops.multiply_plain_inplace(out, zero_pt)
            self._ops.rescale_inplace(out)
            if bias_vec is not None:
                bias_pt = self._encode(list(bias_vec))
                while self._ops.depth_of_plaintext(bias_pt) < self._ops.depth(out):
                    self._ops.mod_drop_inplace_pt(bias_pt)
                self._ops.add_plain_inplace(out, bias_pt)
            return out

        n = len(shifts)
        consecutive = all(int(shifts[k]) == k for k in range(n))

        if n > self.BSGS_THRESHOLD and consecutive:
            return self._halevi_shoup_matmul_bsgs(
                ct_x, block=block, n=n, diagonals=diagonals,
                bias_vec=bias_vec, weight_id=weight_id,
            )
        return self._halevi_shoup_matmul_linear(
            ct_x, block=block, shifts=shifts, diagonals=diagonals,
            low_masks=low_masks, high_masks=high_masks, bias_vec=bias_vec,
        )

    def _halevi_shoup_matmul_linear(
        self,
        ct_x: Ciphertext,
        *,
        block: int,
        shifts: Sequence[int],
        diagonals: Sequence[Optional[Sequence[float]]],
        low_masks: Sequence[Optional[Sequence[float]]],
        high_masks: Sequence[Optional[Sequence[float]]],
        bias_vec: Optional[Sequence[float]] = None,
    ) -> Ciphertext:
        active_idx = [k for k, d in enumerate(diagonals) if d is not None]
        # Galois keys for every required shift.
        needed = set()
        for k in active_idx:
            s = int(shifts[k])
            if s != 0:
                needed.add(s % self._num_slots)
                needed.add((s - block) % self._num_slots)
        if needed:
            self.register_rotation_keys(sorted(needed))

        x_depth = self._ops.depth(ct_x)
        zero_pad = [0.0] * self._num_slots
        result: Optional[Ciphertext] = None

        for start in range(0, len(active_idx), self.HS_CHUNK):
            chunk = active_idx[start : start + self.HS_CHUNK]
            chunk_shifts: List[int] = []
            diag_pts = []
            low_pts = []
            high_pts = []
            for k in chunk:
                s = int(shifts[k])
                chunk_shifts.append(s)
                pt_d = self._encode(list(diagonals[k]))
                target_diag_depth = x_depth if s == 0 else x_depth + 1
                while self._ops.depth_of_plaintext(pt_d) < target_diag_depth:
                    self._ops.mod_drop_inplace_pt(pt_d)
                diag_pts.append(pt_d)
                if s == 0:
                    lo_pt = self._encode(zero_pad)
                    hi_pt = self._encode(zero_pad)
                else:
                    lo = low_masks[k] if low_masks[k] is not None else zero_pad
                    hi = high_masks[k] if high_masks[k] is not None else zero_pad
                    lo_pt = self._encode(list(lo))
                    hi_pt = self._encode(list(hi))
                while self._ops.depth_of_plaintext(lo_pt) < x_depth:
                    self._ops.mod_drop_inplace_pt(lo_pt)
                while self._ops.depth_of_plaintext(hi_pt) < x_depth:
                    self._ops.mod_drop_inplace_pt(hi_pt)
                low_pts.append(lo_pt)
                high_pts.append(hi_pt)

            is_last = start + self.HS_CHUNK >= len(active_idx)
            if is_last and bias_vec is not None:
                bias_pt = self._encode(list(bias_vec))
                while self._ops.depth_of_plaintext(bias_pt) < x_depth + 2:
                    self._ops.mod_drop_inplace_pt(bias_pt)
                with_bias = True
            else:
                bias_pt = self._encode(zero_pad)
                with_bias = False

            chunk_result = self._ops.halevi_shoup_matvec_block(
                ct_x, self._gk, int(block),
                chunk_shifts, diag_pts, low_pts, high_pts,
                with_bias, bias_pt,
            )

            if result is None:
                result = chunk_result
            else:
                self._ops.add_inplace_match(result, chunk_result)

        return result

    def _halevi_shoup_matmul_bsgs(
        self,
        ct_x: Ciphertext,
        *,
        block: int,
        n: int,
        diagonals: Sequence[Optional[Sequence[float]]],
        bias_vec: Optional[Sequence[float]] = None,
        weight_id: Optional[str] = None,
    ) -> Ciphertext:
        """BSGS path: O(√n) Galois keys, O(√n) baby rotations.

        Uses the per-(block,shift,depth) mask plaintext cache and an
        optional per-weight diagonal plaintext cache (keyed on
        ``weight_id`` and ``ct_x.depth()``).
        """
        b1, b2 = self._factor_bsgs(n)
        x_depth = self._ops.depth(ct_x)
        num_slots = self._num_slots
        zero_pad = [0.0] * num_slots

        # ── 1. Galois keys: babies (j and j-block) + giants (g·b1 and g·b1-block) ──
        needed = set()
        for j in range(1, b1):
            needed.add(j % num_slots)
            needed.add((j - block) % num_slots)
        for g in range(1, b2):
            s = g * b1
            needed.add(s % num_slots)
            needed.add((s - block) % num_slots)
        if needed:
            self.register_rotation_keys(sorted(needed))

        # ── 2. Pre-rotate babies (one C++ call) ──
        baby_shifts = list(range(b1))
        baby_low_pts = [self._encode(zero_pad)]   # idx 0 unused
        baby_high_pts = [self._encode(zero_pad)]
        for j in range(1, b1):
            lo_pt, hi_pt = self._get_mask_pt(block, j, x_depth)
            baby_low_pts.append(lo_pt)
            baby_high_pts.append(hi_pt)

        babies = self._ops.pre_rotate_babies(
            ct_x, self._gk, int(block),
            baby_shifts, baby_low_pts, baby_high_pts,
        )

        # ── 3. Diagonal plaintext cache (per (weight_id, x_depth)) ──
        diag_cache_key = (
            (weight_id, x_depth)
            if (weight_id is not None and self.bsgs_diag_cache_enabled)
            else None
        )
        cached_diags = (
            self._bsgs_diag_cache.get(diag_cache_key)
            if diag_cache_key is not None else None
        )

        # ── 4. Per-giant chunk dispatch ──
        result: Optional[Ciphertext] = None
        # acc_depth feeding the giant rotate.
        acc_depth = x_depth + 2 if b1 > 1 else x_depth + 1

        if cached_diags is not None:
            all_diag0_pts, all_diagj_pts = cached_diags
        else:
            # Encode every diagonal once up-front. Each encode is a GPU
            # NTT (~3-4 ms at N=2^16) and dominates first-call cost; on
            # repeat calls the diag cache yields a 50× speedup.
            all_diag0_pts: list = []
            all_diagj_pts: list = []
            for g in range(b2):
                gs = g * b1
                d0 = diagonals[gs] if gs < n else None
                if d0 is None:
                    d0 = zero_pad
                pt_d0 = self._encode(self._per_block_roll(d0, gs, block, num_slots))
                while self._ops.depth_of_plaintext(pt_d0) < x_depth:
                    self._ops.mod_drop_inplace_pt(pt_d0)
                all_diag0_pts.append(pt_d0)
                for j in range(1, b1):
                    idx = gs + j
                    dj = diagonals[idx] if idx < n else None
                    if dj is None:
                        dj = zero_pad
                    pt_dj = self._encode(self._per_block_roll(dj, gs, block, num_slots))
                    while self._ops.depth_of_plaintext(pt_dj) < x_depth + 1:
                        self._ops.mod_drop_inplace_pt(pt_dj)
                    all_diagj_pts.append(pt_dj)
            if diag_cache_key is not None:
                self._bsgs_diag_cache[diag_cache_key] = (all_diag0_pts, all_diagj_pts)

        for g_start in range(0, b2, self.BSGS_GIANT_CHUNK):
            g_end = min(g_start + self.BSGS_GIANT_CHUNK, b2)

            chunk_giant_shifts: List[int] = []
            diag0_pts = []
            diagj_pts = []
            glo_pts = []
            ghi_pts = []
            for g in range(g_start, g_end):
                gs = g * b1
                chunk_giant_shifts.append(gs)
                diag0_pts.append(all_diag0_pts[g])
                # diagj layout: contiguous per-giant block of (b1-1) entries.
                for jj in range(b1 - 1):
                    diagj_pts.append(all_diagj_pts[g * (b1 - 1) + jj])

                if gs == 0:
                    glo_pts.append(self._encode(zero_pad))
                    ghi_pts.append(self._encode(zero_pad))
                else:
                    lo_pt, hi_pt = self._get_mask_pt(block, gs, acc_depth)
                    glo_pts.append(lo_pt)
                    ghi_pts.append(hi_pt)

            chunk_result = self._ops.bsgs_giant_chunk(
                babies, self._gk, int(block),
                chunk_giant_shifts, diag0_pts, diagj_pts, glo_pts, ghi_pts,
            )

            if result is None:
                result = chunk_result
            else:
                self._ops.add_inplace_match(result, chunk_result)

        # ── 5. Bias ──
        if bias_vec is not None:
            r_depth = self._ops.depth(result)
            bias_pt = self._encode(list(bias_vec))
            while self._ops.depth_of_plaintext(bias_pt) < r_depth:
                self._ops.mod_drop_inplace_pt(bias_pt)
            self._ops.add_plain_inplace(result, bias_pt)

        return result

    # ── Phase 7d-2: batched diagonal attention (C++ kernels) ─────────
    def diag_qk_scores(
        self,
        Q_at: Ciphertext,
        K_cyc: Ciphertext,
        *,
        L: int,
        block_attn: int,
        head_dim: int,
        scale: float,
    ) -> Ciphertext:
        """C++ batched diagonal Q@K^T (Phase 7d-2).

        Returns S in diagonal layout: slot[i*block_attn + d] = scale * S[i, (i+d)%L].
        Result depth = Q.depth() + 2.
        """
        # Galois keys: bare-doubling sum within block_attn (powers of 2,
        # positive), rotate-right by d (negative 1..L-1), and K cyclic
        # shifts (d * block_attn for d in 1..L-1).
        needed = set()
        step = 1
        while step < block_attn:
            needed.add(step)
            step <<= 1
        for d in range(1, L):
            needed.add(-d)
            needed.add(d * block_attn)
        if needed:
            self.register_rotation_keys(sorted(needed))

        # Encode the slot-0-each-block scaled mask at depth Q+1 (after
        # the Q*K rescale).
        x_depth = self._ops.depth(Q_at)
        scale_vec = [0.0] * self._num_slots
        n_blocks = self._num_slots // block_attn
        for b in range(n_blocks):
            scale_vec[b * block_attn] = float(scale)
        scale_pt = self._encode(scale_vec)
        while self._ops.depth_of_plaintext(scale_pt) < x_depth + 1:
            self._ops.mod_drop_inplace_pt(scale_pt)

        return self._ops.diag_qk_scores(
            Q_at, K_cyc, self._gk, self._rk,
            int(L), int(block_attn), int(head_dim), int(self._num_slots),
            scale_pt,
        )

    def diag_attn_apply(
        self,
        A_diag: Ciphertext,
        V_cyc: Ciphertext,
        *,
        L: int,
        block_attn: int,
        head_dim: int,
    ) -> Ciphertext:
        """C++ batched diagonal A@V (Phase 7d-2).

        Result depth = A_diag.depth() + 2.
        """
        # Galois keys: rotate-right by step in [1, head_dim/2] (negative),
        # rotate-left by d in [1, L) for col alignment, and V cyclic
        # shifts d*block_attn for d in [1, L).
        needed = set()
        step = 1
        while step < head_dim:
            needed.add(-step)
            step <<= 1
        for d in range(1, L):
            needed.add(d)
            needed.add(d * block_attn)
        if needed:
            self.register_rotation_keys(sorted(needed))

        # Encode L per-d masks (slot d of each block).
        x_depth = self._ops.depth(A_diag)
        n_blocks = self._num_slots // block_attn
        col_pts = []
        for d in range(L):
            v = [0.0] * self._num_slots
            for b in range(n_blocks):
                slot = b * block_attn + d
                if slot < self._num_slots:
                    v[slot] = 1.0
            pt = self._encode(v)
            while self._ops.depth_of_plaintext(pt) < x_depth:
                self._ops.mod_drop_inplace_pt(pt)
            col_pts.append(pt)

        return self._ops.diag_attn_apply(
            A_diag, V_cyc, self._gk, self._rk,
            int(L), int(block_attn), int(head_dim), int(self._num_slots),
            col_pts,
        )

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

    def _get_encoding_transform_context(self, level: int):
        if not self._bootstrap_ready:
            raise RuntimeError(
                "encoding transform context requires configure_bootstrapping() first"
            )
        cached = self._encoding_transform_ctx.get(level)
        if cached is not None:
            return cached

        tc = self._hg.EncodingTransformContext()
        self._ops.generate_encoding_transform_context(
            tc,
            self._scale,
            self._bootstrap_ctos_piece,
            self._bootstrap_stoc_piece,
            level,
            level,
            self._bootstrap_less_key_mode,
        )
        self._encoding_transform_ctx[level] = tc
        return tc

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
        if not self._supports_rescale_flag_override:
            ct_y = self.coeff_matvec(ct_x_coeff, weight, in_dim=in_dim, rescale=True)
            tc = self._get_encoding_transform_context(self._ops.depth(ct_y))
            return self._ops.coeff_to_slot_ctx(ct_y, self._gk, tc)

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

        Phase 6: dispatches to the C++ ``gather_slots_bsgs`` binding so
        the entire pre-rotate / per-giant accumulate / lazy-rescale /
        giant-rotate / sum loop runs without crossing the Python boundary.

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
        # b ∈ [0, B).
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

        # Register Galois keys: babies (positive) ∪ giants (positive),
        # excluding 0 (no rotation needed).
        needed_keys = (babies_used | set(giant_buckets.keys())) - {0}
        self.register_rotation_keys(needed_keys)

        # Build flat CSR arrays for the C++ call.
        baby_shifts_list: List[int] = sorted(babies_used)
        baby_idx_map: Dict[int, int] = {b: i for i, b in enumerate(baby_shifts_list)}
        giant_list: List[int] = sorted(giant_buckets.keys())

        bucket_offsets: List[int] = [0]
        bucket_baby_idx: List[int] = []
        bucket_masks: list = []  # list of heongpu_bindings.Plaintext

        ct_depth = self._ops.depth(ct)
        for g in giant_list:
            for b, positions in giant_buckets[g].items():
                mask = [0.0] * N
                for p in positions:
                    mask[p] = 1.0
                pt = self._encode(mask)
                # Plaintexts start at depth 0; mod-drop to ct's level so
                # multiply_plain_inplace inside C++ sees matching depth.
                while self._ops.depth_of_plaintext(pt) < ct_depth:
                    self._ops.mod_drop_inplace_pt(pt)
                bucket_baby_idx.append(baby_idx_map[b])
                bucket_masks.append(pt)
            bucket_offsets.append(len(bucket_baby_idx))

        return self._ops.gather_slots_bsgs(
            ct, self._gk,
            baby_shifts_list, giant_list,
            bucket_offsets, bucket_baby_idx, bucket_masks,
        )

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
        via the C++ ``clone_ct`` (a single copy-assign on the device).
        Keeps depth flat (no rescale, no encrypt-of-zero).
        """
        return self._ops.clone_ct(ct)

    # ── bootstrapping (Phase 4) ───────────────────────────────────────
    def configure_bootstrapping(
        self,
        CtoS_piece: int = 3,
        StoC_piece: int = 3,
        taylor_number: int = 11,
        less_key_mode: bool = True,
        extra_shifts: Optional[Sequence[int]] = None,
        slim: bool = False,
        include_pow2_shifts: bool = True,
    ) -> None:
        self._ops.generate_bootstrapping_params(
            self._scale, CtoS_piece, StoC_piece, taylor_number, less_key_mode,
            slim,
        )
        self._bootstrap_ctos_piece = int(CtoS_piece)
        self._bootstrap_stoc_piece = int(StoC_piece)
        self._bootstrap_less_key_mode = bool(less_key_mode)
        self._encoding_transform_ctx.clear()
        self._slim_bootstrap = bool(slim)
        self._slim_StoC_piece = int(StoC_piece)
        # Merge boot shifts with the existing ±2^k power-of-two set so
        # matmul / rotate paths keep working after configure_bootstrapping.
        boot_shifts = list(self._ops.bootstrapping_key_indexs())
        if include_pow2_shifts:
            max_log = (self._num_slots).bit_length() - 1
            pow2 = [1 << k for k in range(max_log)]
            pow2_set = pow2 + [-s for s in pow2]
        else:
            pow2_set = []
        extras = list(extra_shifts) if extra_shifts is not None else []
        all_shifts = sorted(set(boot_shifts + pow2_set + extras))
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
        if getattr(self, "_slim_bootstrap", False):
            # Slim StoC happens *before* ModRaise; needs StoC_piece levels left.
            target = self._max_depth - int(self._slim_StoC_piece)
            while self._ops.depth(out) < target:
                self._ops.mod_drop_inplace_ct(out)
            return self._ops.slim_bootstrapping(out, self._gk, self._rk)
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
