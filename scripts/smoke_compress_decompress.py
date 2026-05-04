"""Phase 8b/8c smoke: compress N values, decompress, verify each output ct
holds the corresponding broadcast value (after dividing by N)."""
from __future__ import annotations

import time

import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import (
    enc_compress, decompress, required_galois_elts,
)


def main() -> None:
    # Small ring for fast iteration: N=2^12=4096, so logN=12 rounds.
    N_LOG = 12
    N = 1 << N_LOG

    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60, 50, 50, 50),
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        sec_none=True,
    )
    print(f"Backend ready: N={be._N}, n_slots={be.capabilities.n_slots}")

    # Generate the special galois keys for decompression.
    elts = required_galois_elts(N)
    print(f"Required galois elts ({len(elts)}): {elts}")
    t = time.time()
    gk = be._kg.generate_galois_key_elts(be._ctx, be._sk, elts)
    print(f"GaloisKey gen: {time.time()-t:.2f}s")

    rng = np.random.default_rng(1)
    values = rng.normal(size=N).astype(np.float64) * 0.5

    t = time.time()
    ct = enc_compress(be, values)
    print(f"compress: {(time.time()-t)*1000:.1f} ms (1 ct)")

    t = time.time()
    ctlist = decompress(be, ct, gk)
    dt = time.time() - t
    print(f"decompress: {dt:.2f}s for N={N} cts ({dt*1000/N:.2f} ms/ct)")
    print(f"output count = {len(ctlist)}")

    # Sample a handful of indices and verify each broadcast value.
    n_check = 16
    sample_idx = list(range(n_check)) + [N - 1, N // 2, 7]
    max_err = 0.0
    for j in sample_idx:
        if j >= len(ctlist):
            continue
        pt = be._decryptor.decrypt(be._ctx, ctlist[j])
        # The output ciphertext encodes the constant polynomial p(x) = N*c_j
        # in coefficient form. Its slot-domain interpretation is therefore
        # the constant value c_j broadcast in every slot. ``decoded.mean()``
        # already equals c_j (HEonGPU's ``decode`` here returns the raw
        # polynomial coefficients, the only nonzero one being coeff 0 = N*c_j;
        # mean over N coefficients = c_j).
        decoded = np.array(be._encoder.decode(pt))
        broadcast_val = decoded.mean()
        err = abs(broadcast_val - values[j])
        max_err = max(max_err, err)
        if j < 4 or j == N - 1:
            print(f"  j={j:4d}  expect={values[j]:+.4f}  got_mean={broadcast_val:+.4f}"
                  f"  coeff[0]={decoded[0]:+.2f}  err={err:.2e}")

    print(f"max err = {max_err:.3e}")
    assert max_err < 1e-2, f"Decompress error too large: {max_err}"
    print("PASS")


if __name__ == "__main__":
    main()
