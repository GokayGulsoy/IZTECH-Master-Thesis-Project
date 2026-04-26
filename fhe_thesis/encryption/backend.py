"""CKKS backend abstraction.

Phase-1 goal: keep the protocol code (`ops.py`, `packing.py`) free of
direct TenSEAL imports so that a GPU backend (Phantom-FHE, OpenFHE-CUDA,
…) can be plugged in later without touching the protocol layer.

Only the operations actually exercised by the LPAN protocol are exposed
here. Anything more general belongs in the backend's own module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


# A backend-opaque ciphertext handle. We deliberately type it as `Any`
# so the protocol layer never depends on a concrete library type.
Ciphertext = Any


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can / cannot do, used by depth.py and ops.py."""

    name: str
    supports_bootstrapping: bool
    supports_galois_rotations: bool
    n_slots: int  # max real slots = poly_modulus_degree / 2
    initial_levels: int  # length of coeff-mod chain minus the two 60-bit primes


class CKKSBackend(ABC):
    """Minimal CKKS surface used by the LPAN protocol.

    All vector arguments are plain Python lists of floats; the backend
    is responsible for any internal conversion. Returned ciphertexts
    are opaque handles consumed only by other backend methods.
    """

    capabilities: BackendCapabilities

    # ── encoding / I/O ────────────────────────────────────────────────
    @abstractmethod
    def encrypt(self, values: Sequence[float]) -> Ciphertext: ...

    @abstractmethod
    def decrypt(self, ct: Ciphertext) -> List[float]: ...

    # ── linear ops on ciphertexts ─────────────────────────────────────
    @abstractmethod
    def add(self, a: Ciphertext, b: Ciphertext) -> Ciphertext: ...

    @abstractmethod
    def sub(self, a: Ciphertext, b: Ciphertext) -> Ciphertext: ...

    @abstractmethod
    def add_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext: ...

    @abstractmethod
    def mul_plain(self, a: Ciphertext, plain: Sequence[float]) -> Ciphertext: ...

    @abstractmethod
    def mul(self, a: Ciphertext, b: Ciphertext) -> Ciphertext: ...

    # ── polynomial evaluation ─────────────────────────────────────────
    @abstractmethod
    def polyval(self, ct: Ciphertext, power_coeffs: Sequence[float]) -> Ciphertext:
        """Evaluate Σ c_i · x^i on a ciphertext (coeffs in power basis)."""

    # ── matmul (token vector · weight matrix) ─────────────────────────
    @abstractmethod
    def matmul_plain(
        self,
        ct: Ciphertext,
        weight: Sequence[Sequence[float]],
        bias: Optional[Sequence[float]] = None,
    ) -> Ciphertext:
        """Compute (W · x) + b where x is encrypted, W and b are plain.

        `weight` is shape (out_dim, in_dim) row-major.
        """

    # ── attention primitives (Phase 2) ────────────────────────────────
    @abstractmethod
    def dot(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        """Inner product ⟨a, b⟩. Returned ct holds the scalar in slot 0
        (and may broadcast it across slots, depending on the backend)."""

    def broadcast_first_slot(
        self, ct: Ciphertext, n: int, scale: float = 1.0
    ) -> Ciphertext:
        """Return a ct whose first n slots all hold ``scale * slot0(ct)``.

        Default implementation uses ``matmul_plain`` with an (n, 1) column
        vector of ``scale`` — works on backends where size-1 cts are first-class
        (e.g. TenSEAL after .sum()). Backends with full-width broadcast scalars
        (e.g. OpenFHE EvalSum) can override with a single ``mul_plain``.
        """
        return self.matmul_plain(ct, [[scale]] * n)

    def place_scaled_at_slot(
        self, ct: Ciphertext, slot: int, n: int, scale: float = 1.0
    ) -> Ciphertext:
        """Return a length-n ct where slot[``slot``] = ``scale·slot0(ct)`` and others 0.

        Used by attention to scatter scalar dot-products into per-row score
        vectors. Default uses ``matmul_plain`` with a sparse (n, 1) column;
        backends with full-width broadcast scalars override with a one-hot
        ``mul_plain`` mask.
        """
        weight = [[0.0]] * n
        weight[slot] = [scale]
        return self.matmul_plain(ct, weight)

    def rotate(self, ct: Ciphertext, steps: int) -> Ciphertext:
        """Rotate ciphertext slots left by *steps* (negative = right).

        Override in backends that support Galois rotations.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support rotate")

    @abstractmethod
    def sum_slots(self, ct: Ciphertext) -> Ciphertext:
        """Sum of all slots, broadcast back to slot 0 (scalar ct)."""


# ──────────────────────────────────────────────────────────────────────
# Reference backend: TenSEAL
# ──────────────────────────────────────────────────────────────────────


class TenSEALBackend(CKKSBackend):
    """TenSEAL-backed CKKS implementation.

    Constructed lazily so that environments without TenSEAL installed
    (e.g. the design machine) can still import this module.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 16384,
        coeff_mod_bit_sizes: Optional[List[int]] = None,
        global_scale_bits: int = 40,
    ) -> None:
        import tenseal as ts  # local import: heavy dep

        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60] + [40] * 6 + [60]

        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        ctx.global_scale = 2**global_scale_bits
        ctx.generate_galois_keys()

        self._ts = ts
        self._ctx = ctx
        self.capabilities = BackendCapabilities(
            name="tenseal",
            supports_bootstrapping=False,
            supports_galois_rotations=True,
            n_slots=poly_modulus_degree // 2,
            initial_levels=len(coeff_mod_bit_sizes) - 2,
        )

    # ── encoding / I/O ────────────────────────────────────────────────
    def encrypt(self, values):
        return self._ts.ckks_vector(self._ctx, list(values))

    def decrypt(self, ct):
        return list(ct.decrypt())

    # ── arithmetic ────────────────────────────────────────────────────
    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def add_plain(self, a, plain):
        return a + list(plain)

    def mul_plain(self, a, plain):
        return a * list(plain)

    def mul(self, a, b):
        return a * b

    # ── polynomial / matmul ───────────────────────────────────────────
    def polyval(self, ct, power_coeffs):
        return ct.polyval(list(power_coeffs))

    def matmul_plain(self, ct, weight, bias=None):
        # TenSEAL's CKKSVector.matmul expects (in_dim, out_dim), so we
        # transpose. weight here is row-major (out_dim, in_dim).
        wT = [[row[j] for row in weight] for j in range(len(weight[0]))]
        out = ct.matmul(wT)
        if bias is not None:
            out = out + list(bias)
        return out

    # ── attention primitives ──────────────────────────────────────────
    def dot(self, a, b):
        return a.dot(b)

    def sum_slots(self, ct):
        return ct.sum()
