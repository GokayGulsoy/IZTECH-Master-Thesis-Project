"""Smoke test for newly-added HEonGPUBackend ops.

Validates polyval (Horner), matmul_plain (Halevi-Shoup), dot, and
sum_slots against numpy ground truth on the H100 wrapper.
"""
from __future__ import annotations

import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main() -> None:
    rng = np.random.default_rng(0)
    backend = HEonGPUBackend(poly_modulus_degree=2**14, sec_none=True)
    n = backend.num_slots

    # ── polyval (deg 4): p(x) = 1 - 0.5 x + 0.25 x^2 - 0.1 x^3 + 0.01 x^4 ──
    coeffs = [1.0, -0.5, 0.25, -0.1, 0.01]
    x = rng.uniform(-1.0, 1.0, n).astype(np.float64)
    ct = backend.encrypt(x.tolist())
    pv = backend.polyval(ct, coeffs)
    got = np.asarray(backend.decrypt(pv))[:n]
    expect = sum(c * x ** i for i, c in enumerate(coeffs))
    err = np.max(np.abs(got - expect))
    print(f"[polyval ] deg=4 max-err = {err:.3e}")
    assert err < 1e-3, "polyval mismatch"

    # ── sum_slots ──────────────────────────────────────────────────────
    s = backend.sum_slots(ct)
    got = np.asarray(backend.decrypt(s))[0]
    err = abs(got - x.sum())
    print(f"[sum_slot] err = {err:.3e}  (expected ≈ {x.sum():.3f})")
    assert err < 1e-2

    # ── dot ────────────────────────────────────────────────────────────
    y = rng.uniform(-1.0, 1.0, n).astype(np.float64)
    ct_y = backend.encrypt(y.tolist())
    d = backend.dot(ct, ct_y)
    got = np.asarray(backend.decrypt(d))[0]
    err = abs(got - float(np.dot(x, y)))
    print(f"[dot     ] err = {err:.3e}  (expected ≈ {np.dot(x, y):.3f})")
    assert err < 1e-2

    # ── matmul_plain ───────────────────────────────────────────────────
    in_dim, out_dim = 32, 16
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float64) * 0.1
    b = rng.standard_normal(out_dim).astype(np.float64) * 0.1
    xv = rng.uniform(-1.0, 1.0, in_dim).astype(np.float64)
    padded = np.zeros(n)
    padded[:in_dim] = xv
    ct_x = backend.encrypt(padded.tolist())
    mm = backend.matmul_plain(ct_x, W.tolist(), b.tolist())
    got = np.asarray(backend.decrypt(mm))[:out_dim]
    expect = W @ xv + b
    err = np.max(np.abs(got - expect))
    print(f"[matmul  ] {out_dim}x{in_dim}  max-err = {err:.3e}")
    assert err < 1e-3

    print("\n✅ All HEonGPU token-path ops PASS")


if __name__ == "__main__":
    main()
