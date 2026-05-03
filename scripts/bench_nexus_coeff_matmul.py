"""Benchmark NEXUS-style coefficient-packed matmul vs Halevi-Shoup on H100.

Compares :meth:`HEonGPUBackend.coeff_matvec` (1 multiplication for the
whole matvec) against the existing :meth:`matmul_plain` (Halevi-Shoup,
``next_pow2(in_dim)`` rotations + multiplies) on a BERT-tiny FFN-shaped
linear (128 → 512). Demonstrates the predicted ~100× speedup and
validates correctness against numpy.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main() -> int:
    in_dim, out_dim = 128, 512  # BERT-tiny W1 shape
    print(f"shape: in={in_dim} out={out_dim}  m·n = {in_dim * out_dim}")

    print("\nInit HEonGPU N=2^16 (no bootstrap, fast smoke)...")
    t0 = time.time()
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60, 50, 50, 50, 50, 60),
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    print(f"  keygen: {time.time() - t0:.2f}s   N={be._N}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim).astype(np.float64) * 0.3
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float64) * 0.05
    expected = W @ x

    # ── Existing path: Halevi-Shoup matmul_plain (slot-packed) ─────
    print("\n[old] matmul_plain (Halevi-Shoup)")
    padded = np.zeros(be.num_slots)
    padded[:in_dim] = x
    ct_x_slot = be.encrypt(padded.tolist())
    t = time.time()
    ct_y = be.matmul_plain(ct_x_slot, W.tolist())
    dt_old = time.time() - t
    got_old = np.asarray(be.decrypt(ct_y))[:out_dim]
    err_old = np.max(np.abs(got_old - expected))
    print(f"  wall: {dt_old:.3f}s")
    print(f"  max-err: {err_old:.3e}")

    # ── New path: NEXUS coeff_matvec (1 multiplication) ───────────
    print("\n[new] coeff_matvec (NEXUS)")
    t = time.time()
    ct_x_coeff = be.encrypt_coeff(x.tolist())
    t_enc = time.time() - t
    t = time.time()
    ct_yn = be.coeff_matvec(ct_x_coeff, W, in_dim=in_dim)
    dt_new = time.time() - t
    got_new = np.asarray(be.decrypt_coeff_extract(ct_yn, in_dim, out_dim))
    err_new = np.max(np.abs(got_new - expected))
    print(f"  encrypt_coeff: {t_enc:.3f}s")
    print(f"  matvec wall:   {dt_new:.3f}s")
    print(f"  max-err:       {err_new:.3e}")

    print()
    print(f"speedup:  {dt_old / dt_new:.1f}×   ({dt_old:.3f}s → {dt_new:.3f}s)")
    return 0 if (err_new < 5e-3 and err_old < 5e-3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
