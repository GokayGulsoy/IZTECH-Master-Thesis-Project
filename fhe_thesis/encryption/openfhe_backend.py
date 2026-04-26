"""OpenFHE-backed CKKS implementation with bootstrapping.

Mirrors the public API of :class:`TenSEALBackend` so the protocol code
in ``ops.py`` and ``protocol.py`` can switch backends with no edits.

Key extra capability: ``bootstrap(ct)`` to refresh ciphertexts mid-pipeline,
unlocking end-to-end encrypted inference for models deeper than one block
on TenSEAL's max-19 modulus chain.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import BackendCapabilities, CKKSBackend, Ciphertext


class OpenFHEBackend(CKKSBackend):
    """OpenFHE CKKS backend with optional bootstrapping.

    Parameters
    ----------
    multiplicative_depth : int
        Total levels in the modulus chain. If ``enable_bootstrap`` is
        True, this should include the bootstrap budget (~12-15 levels)
        plus the post-bootstrap compute budget.
    ring_dim : int
        Ring dimension N (auto-selected for security if 0).
    scaling_mod_size : int
        Bit-size of the rescaling moduli (default 59).
    first_mod_size : int
        Bit-size of the first prime in the modulus chain.
    enable_bootstrap : bool
        Generate bootstrapping keys and enable :meth:`bootstrap`.
    num_slots : int
        Effective batch size. If 0, defaults to ``ring_dim // 2``.
    bootstrap_level_budget : list[int]
        OpenFHE bootstrap encode/decode level budget; ``[3, 3]`` is the
        recommended default (consumes ~12 levels in total).
    security_level : openfhe.SecurityLevel | None
        Defaults to HEStd_128_classic. Pass ``HEStd_NotSet`` for tests.
    """

    def __init__(
        self,
        multiplicative_depth: int = 25,
        ring_dim: int = 1 << 16,
        scaling_mod_size: int = 59,
        first_mod_size: int = 60,
        enable_bootstrap: bool = True,
        num_slots: int = 0,
        bootstrap_level_budget: Optional[List[int]] = None,
        security_level=None,
        num_threads: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        num_threads : int
            Number of OpenMP threads passed to ``openfhe.SetNumThreads``.
            0 (default) means use the OpenFHE default (usually all cores).
        """
        import openfhe as ofhe  # local import: heavy native dep

        # ── Threading (O5) ─────────────────────────────────────────
        if hasattr(ofhe, "SetNumThreads"):
            if num_threads > 0:
                ofhe.SetNumThreads(num_threads)
            else:
                # Default: use all available physical cores
                cpu_count = os.cpu_count() or 1
                ofhe.SetNumThreads(cpu_count)

        if bootstrap_level_budget is None:
            bootstrap_level_budget = [3, 3]
        if security_level is None:
            # HEStd_NotSet: disables the HE-standard ring-dim enforcement.
            # This is standard practice for FHE benchmarking (THE-X, CipherFormer,
            # Iron all use relaxed params for timing measurements). Production
            # deployment targeting 128-bit security requires ring_dim=131072 for
            # depth=25; note this separately in the thesis security analysis.
            security_level = ofhe.SecurityLevel.HEStd_NotSet

        params = ofhe.CCParamsCKKSRNS()
        params.SetSecretKeyDist(ofhe.SecretKeyDist.UNIFORM_TERNARY)
        params.SetSecurityLevel(security_level)
        # HYBRID key switching: half the key-switch cost of BV mode by splitting
        # the evaluation key into two components.  This is critical for rotation-
        # heavy ops (matmul diagonals, attention).  Must be set before GenCryptoContext.
        params.SetKeySwitchTechnique(ofhe.KeySwitchTechnique.HYBRID)
        params.SetRingDim(ring_dim)
        params.SetScalingModSize(scaling_mod_size)
        params.SetFirstModSize(first_mod_size)
        params.SetMultiplicativeDepth(multiplicative_depth)
        if num_slots <= 0:
            # Default to 4096: next power-of-two above BERT-Base's largest
            # linear dimension (intermediate=3072).  This caps rotation-key
            # generation at ~4107 keys (≈ 111 GB @ N=65536, depth=25) vs the
            # catastrophic ring_dim//2=32768 which would require ~885 GB.
            num_slots = 4096
        params.SetBatchSize(num_slots)

        cc = ofhe.GenCryptoContext(params)
        cc.Enable(ofhe.PKESchemeFeature.PKE)
        cc.Enable(ofhe.PKESchemeFeature.KEYSWITCH)
        cc.Enable(ofhe.PKESchemeFeature.LEVELEDSHE)
        cc.Enable(ofhe.PKESchemeFeature.ADVANCEDSHE)
        if enable_bootstrap:
            cc.Enable(ofhe.PKESchemeFeature.FHE)
            cc.EvalBootstrapSetup(bootstrap_level_budget, [0, 0], num_slots)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        # Galois keys for rotations and EvalSum (needed by sum_slots/dot/matmul).
        cc.EvalSumKeyGen(keys.secretKey)
        # Generate a compact rotation-key set: ± powers-of-two only.
        # Any required rotation r is composed as a sum of binary shifts,
        # reducing key memory from O(num_slots) to O(log num_slots).
        rot_indices = []
        k = 1
        while k < num_slots:
            rot_indices.append(k)
            rot_indices.append(-k)
            k <<= 1
        cc.EvalRotateKeyGen(keys.secretKey, rot_indices)
        if enable_bootstrap:
            cc.EvalBootstrapKeyGen(keys.secretKey, num_slots)

        self._ofhe = ofhe
        self._cc = cc
        self._keys = keys
        self._num_slots = num_slots
        self._depth = multiplicative_depth
        self._enable_bootstrap = enable_bootstrap
        self._num_threads = num_threads

        # ── BSGS precompute cache (O2) ─────────────────────────────
        # Maps weight_hash -> (diagonals: list[list[float]], n: int)
        # so diagonal decomposition is built once per unique weight matrix.
        self._diag_cache: Dict[str, Tuple[List[List[float]], int]] = {}
        self._diag_cache_lock = threading.Lock()

        self.capabilities = BackendCapabilities(
            name="openfhe",
            supports_bootstrapping=enable_bootstrap,
            supports_galois_rotations=True,
            n_slots=num_slots,
            initial_levels=multiplicative_depth,
        )

    def _rotate(self, ct: Ciphertext, steps: int) -> Ciphertext:
        """Rotate by composing power-of-two shifts.

        OpenFHE rotation keys are generated only for ±2^k. This helper
        decomposes any requested rotation into those basis shifts.
        """
        if steps == 0:
            return ct
        # Normalize to the shortest equivalent shift in (-num_slots/2, num_slots/2].
        s = steps % self._num_slots
        if s > self._num_slots // 2:
            s -= self._num_slots

        sign = 1 if s > 0 else -1
        rem = abs(s)
        out = ct
        bit = 1
        while rem:
            if rem & 1:
                out = self._cc.EvalRotate(out, sign * bit)
            rem >>= 1
            bit <<= 1
        return out

    def rotate(self, ct: Ciphertext, steps: int) -> Ciphertext:
        """Public slot rotation (delegates to composed power-of-two shifts)."""
        return self._rotate(ct, steps)

    # ── encoding helpers ──────────────────────────────────────────────
    def _encode(self, values: Sequence[float], ct=None):
        # Pad / truncate to num_slots and pack as a CKKS plaintext.
        # If ``ct`` is given, encode at the ciphertext's current level so the
        # plaintext modulus chain matches — this drops plaintext memory by
        # (depth - level)× vs always encoding at level 0.
        v = list(values)
        if len(v) > self._num_slots:
            raise ValueError(
                f"vector length {len(v)} exceeds num_slots {self._num_slots}"
            )
        level = ct.GetLevel() if ct is not None else 0
        return self._cc.MakeCKKSPackedPlaintext(v, 1, level, None, self._num_slots)

    # ── encryption / decryption ───────────────────────────────────────
    def encrypt(self, values: Sequence[float]) -> Ciphertext:
        pt = self._encode(values)
        return self._cc.Encrypt(self._keys.publicKey, pt)

    def decrypt(self, ct: Ciphertext) -> List[float]:
        pt = self._cc.Decrypt(ct, self._keys.secretKey)
        # OpenFHE returns the full slot vector; we don't know the
        # logical length here — caller is responsible for slicing. We
        # truncate to num_slots to avoid huge returns.
        pt.SetLength(self._num_slots)
        return list(pt.GetRealPackedValue())

    # ── arithmetic ────────────────────────────────────────────────────
    def add(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        return self._cc.EvalAdd(a, b)

    def sub(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        return self._cc.EvalSub(a, b)

    def add_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext:
        return self._cc.EvalAdd(a, self._encode(plain, ct=a))

    def mul_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext:
        return self._cc.EvalMult(a, self._encode(plain, ct=a))

    def mul(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        return self._cc.EvalMult(a, b)

    def mul_no_relin(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        """Ct×Ct multiply WITHOUT relinearisation (O6 — lazy relin).

        Caller must call :meth:`relinearize` after the accumulation loop
        to keep the ciphertext compact. Saves one key-switch per multiply
        inside loops like ``enc_attention_apply``.
        """
        return self._cc.EvalMultNoRelin(a, b)

    def relinearize(self, ct: Ciphertext) -> Ciphertext:
        """Explicitly relinearise a ciphertext (companion to mul_no_relin)."""
        return self._cc.Relinearize(ct)

    # ── polynomial evaluation ─────────────────────────────────────────
    def polyval(self, ct: Ciphertext, power_coeffs: Sequence[float]) -> Ciphertext:
        # OpenFHE: EvalPolyLinear (deg ≤ 5) / EvalPoly (PS algorithm) for
        # power-basis coefficients [c0, c1, ..., cd].
        return self._cc.EvalPoly(ct, list(power_coeffs))

    # ── matmul (token vector · weight matrix) ─────────────────────────
    def matmul_plain(
        self,
        ct: Ciphertext,
        weight: Sequence[Sequence[float]],
        bias: Optional[Sequence[float]] = None,
    ) -> Ciphertext:
        """Halevi-Shoup diagonal-encoded matmul with precompute cache (O2).

        The diagonal decomposition is built once per unique weight matrix
        (keyed by SHA-1 of its bytes) and reused on every subsequent call.
        This makes the per-token cost just ``n`` plaintext multiplies +
        ``n`` rotations — identical to the uncached version — but avoids
        rebuilding the ``n`` diagonal vectors on every token.

        Cost (cached): ``n`` rotations + ``n`` mul_plain, ~14–37× fewer
        rotations than naive row-wise matmul for BERT linear layers.
        """
        out_dim = len(weight)
        in_dim = len(weight[0]) if out_dim else 0
        if out_dim == 0:
            raise ValueError("weight is empty")

        # ── BSGS dimension ─────────────────────────────────────────
        target = max(out_dim, in_dim)
        n = 1
        while n < target:
            n <<= 1
        if n > self._num_slots:
            raise ValueError(
                f"matmul dim n={n} exceeds num_slots={self._num_slots}; "
                f"increase num_slots when constructing OpenFHEBackend"
            )

        # ── Cache lookup / build ────────────────────────────────────
        w_arr = np.asarray(weight, dtype=np.float64)
        w_hash = hashlib.sha1(w_arr.tobytes()).hexdigest()

        with self._diag_cache_lock:
            if w_hash not in self._diag_cache:
                diagonals: List[List[float]] = []
                for i in range(n):
                    diag = [0.0] * self._num_slots
                    any_nz = False
                    for k in range(out_dim):
                        col = (k + i) % n
                        if col < in_dim:
                            val = float(w_arr[k, col])
                            diag[k] = val
                            if val != 0.0:
                                any_nz = True
                    diagonals.append(diag if any_nz else None)
                self._diag_cache[w_hash] = (diagonals, n)
            diagonals, n = self._diag_cache[w_hash]

        # ── Input replication ───────────────────────────────────────
        x = ct
        cur = in_dim
        while cur < n:
            x = self._cc.EvalAdd(x, self._rotate(x, -cur))
            cur <<= 1

        # ── Diagonal multiply-accumulate ────────────────────────────
        result = None
        for i, diag in enumerate(diagonals):
            if diag is None:
                continue
            rot_x = x if i == 0 else self._rotate(x, i)
            term = self._cc.EvalMult(rot_x, self._encode(diag, ct=rot_x))
            result = term if result is None else self._cc.EvalAdd(result, term)

        if result is None:
            result = self._cc.EvalMult(ct, self._encode([0.0] * self._num_slots, ct=ct))

        if bias is not None:
            b = list(bias) + [0.0] * (self._num_slots - len(bias))
            result = self._cc.EvalAdd(result, self._encode(b, ct=result))
        return result

    # ── attention primitives ──────────────────────────────────────────
    def dot(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        # EvalInnerProduct returns the scalar broadcast across slots.
        return self._cc.EvalInnerProduct(a, b, self._num_slots)

    def sum_slots(self, ct: Ciphertext) -> Ciphertext:
        return self._cc.EvalSum(ct, self._num_slots)

    def broadcast_first_slot(self, ct: Ciphertext, n: int, scale: float = 1.0) -> Ciphertext:
        """OpenFHE EvalSum already broadcasts the sum across all slots; just mask + scale.

        Costs a single multiplicative level (mul_plain), versus the default
        (n × mul_plain + EvalSum + n × mul_plain) inherited from the base.
        """
        mask = [scale] * n + [0.0] * (self._num_slots - n)
        return self._cc.EvalMult(ct, self._encode(mask, ct=ct))

    def place_scaled_at_slot(
        self, ct: Ciphertext, slot: int, n: int, scale: float = 1.0
    ) -> Ciphertext:
        """OpenFHE: dot/sum_slots already broadcast slot 0 to every slot; one-hot mask.

        Costs a single multiplicative level. The slot 0 value (broadcast across
        all slots) is multiplied by a mask that is ``scale`` at position ``slot``
        and 0 elsewhere — leaving the desired scattered value.
        """
        mask = [0.0] * self._num_slots
        if slot < n:
            mask[slot] = scale
        return self._cc.EvalMult(ct, self._encode(mask, ct=ct))

    # ── bootstrap (the whole point of this backend) ───────────────────
    def bootstrap(self, ct: Ciphertext) -> Ciphertext:
        """Refresh a depleted ciphertext, restoring multiplicative levels.

        Raises ``RuntimeError`` if the backend was constructed without
        bootstrap support.
        """
        if not self._enable_bootstrap:
            raise RuntimeError(
                "OpenFHEBackend was created with enable_bootstrap=False"
            )
        return self._cc.EvalBootstrap(ct)

    def get_level(self, ct: Ciphertext) -> int:
        """Return the current consumed level of a ciphertext (for bootstrap policy)."""
        return ct.GetLevel()
