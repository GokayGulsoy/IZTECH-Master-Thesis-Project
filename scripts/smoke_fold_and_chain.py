"""Phase 8e smoke: fold linear outputs back into a packed ct, run a 2-layer
linear chain via the NEXUS-style packing primitives."""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import (
    enc_compress, linear_compressed, fold_outputs_to_packed,
)


def main() -> None:
    N_LOG = 12
    N = 1 << N_LOG

    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60, 50, 50, 50, 50, 50),
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        sec_none=True,
    )
    print(f"Backend ready: N={be._N}")
    rng = np.random.default_rng(7)

    # ---------------- 1-layer fold sanity ----------------
    M1 = 16
    x = rng.normal(size=N).astype(np.float64) * 0.3
    W1 = rng.normal(size=(M1, N)).astype(np.float64) * 0.05
    expected = W1 @ x

    ct_x = enc_compress(be, x)
    out = linear_compressed(be, W1, ct_x)
    t = time.time()
    packed = fold_outputs_to_packed(be, out)
    fold_dt = time.time() - t
    print(f"fold_outputs_to_packed: {fold_dt*1000:.1f}ms for M={M1}")

    pt = be._decryptor.decrypt(be._ctx, packed)
    coeffs = np.array(be._encoder.decode_coeff(pt))
    err = np.max(np.abs(coeffs[:M1] - expected))
    print(f"  packed coeff err: {err:.2e}  (other-coeffs max abs: {np.max(np.abs(coeffs[M1:])):.2e})")
    assert err < 1e-2, f"fold packing err too large: {err}"
    print("FOLD PASS")

    # ---------------- 2-layer chain ----------------
    print()
    print("=== 2-layer chain ===")
    M2 = 32
    W2 = rng.normal(size=(M2, N)).astype(np.float64) * 0.05
    expected2_full = W2 @ np.concatenate([expected, np.zeros(N - M1)])
    expected2 = expected2_full  # length M2

    # First layer output is "packed" — pad with zeros to length N for input.
    t = time.time()
    out2 = linear_compressed(be, W2, packed)
    L2_dt = time.time() - t
    print(f"layer 2 linear_compressed: {L2_dt*1000:.1f}ms for M={M2}")

    max_err = 0.0
    for i in range(min(8, M2)):
        pt_i = be._decryptor.decrypt(be._ctx, out2[i])
        c = np.array(be._encoder.decode_coeff(pt_i))[0]
        err = abs(c - expected2[i])
        max_err = max(max_err, err)
        print(f"  i={i}  expect={expected2[i]:+.4f}  got={c:+.4f}  err={err:.2e}")
    print(f"max err (layer 2): {max_err:.3e}")
    assert max_err < 1e-2, f"chain err too large: {max_err}"
    print("CHAIN PASS")

    # ---------------- BERT-base scale: 768 -> 768 -> 768 ----------------
    print()
    print("=== BERT scale: 3-layer chain 768 -> 768 -> 768 ===")
    Mb = 768
    Wa = rng.normal(size=(Mb, N)).astype(np.float64) * 0.02
    Wb = rng.normal(size=(Mb, N)).astype(np.float64) * 0.02
    Wc = rng.normal(size=(Mb, N)).astype(np.float64) * 0.02

    t0 = time.time()
    out_a = linear_compressed(be, Wa, ct_x)
    t1 = time.time()
    pkt_a = fold_outputs_to_packed(be, out_a)
    t2 = time.time()
    out_b = linear_compressed(be, Wb, pkt_a)
    t3 = time.time()
    pkt_b = fold_outputs_to_packed(be, out_b)
    t4 = time.time()
    out_c = linear_compressed(be, Wc, pkt_b)
    t5 = time.time()

    print(f"L1 linear: {(t1-t0)*1000:.0f}ms  fold: {(t2-t1)*1000:.0f}ms")
    print(f"L2 linear: {(t3-t2)*1000:.0f}ms  fold: {(t4-t3)*1000:.0f}ms")
    print(f"L3 linear: {(t5-t4)*1000:.0f}ms")
    print(f"TOTAL 3-layer 768x4096: {(t5-t0)*1000:.0f}ms = {(t5-t0):.3f}s")

    # spot-check correctness of layer 3 output
    a_full = np.zeros(N); a_full[:Mb] = Wa @ x
    b_full = np.zeros(N); b_full[:Mb] = Wb @ a_full
    expected_c = (Wc @ b_full)[:8]
    for i in range(8):
        pt_i = be._decryptor.decrypt(be._ctx, out_c[i])
        c = np.array(be._encoder.decode_coeff(pt_i))[0]
        print(f"  i={i}  expect={expected_c[i]:+.4f}  got={c:+.4f}  err={abs(c - expected_c[i]):.2e}")


if __name__ == "__main__":
    main()
