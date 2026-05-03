"""Path A: matrix-packed → bit-rev permute → StoC → nexus_linear → place back.

Validates the full per-token pipeline AND measures real wall + depth cost.
This is the gating experiment for whether path A actually wins e2e.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    print("Init HEonGPU N=2^16...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()

    rng = np.random.default_rng(42)
    seq_len = 8
    hidden = 128
    out_dim = 128
    block = 128
    log_n = (be._num_slots).bit_length() - 1  # 15

    X = rng.standard_normal((seq_len, hidden)) * 0.3
    W = rng.standard_normal((out_dim, hidden)) * 0.05
    b_v = rng.standard_normal(out_dim) * 0.1
    expected_Y = X @ W.T + b_v

    flat = np.zeros(be._num_slots)
    for t in range(seq_len):
        flat[t*block:(t+1)*block] = X[t]
    ct_mp = be.encrypt(flat.tolist())

    def bitrev(i, bits=log_n):
        r = 0
        for k in range(bits):
            r = (r << 1) | ((i >> k) & 1)
        return r

    # ---- Process token 0 only first to validate ----
    print("\n--- TOKEN 0 ---")
    t = 0

    # Step 1: extract + bit-rev permute via gather_slots
    src1 = [t*block + j for j in range(hidden)]
    dst1 = [bitrev(j) for j in range(hidden)]
    # Warmup: registers Galois keys for the bit-rev permute shifts
    _ = be.gather_slots(ct_mp, src1, dst_indices=dst1)
    times = []
    for _ in range(3):
        t0 = time.time()
        ct_perm = be.gather_slots(ct_mp, src1, dst_indices=dst1)
        times.append(time.time() - t0)
    t_perm1 = float(np.median(times))
    print(f"  permute_in (steady):  {t_perm1*1000:.1f}ms  depth_after={be._ops.depth(ct_perm)}")

    # Quick sanity: decrypt and confirm slot[bit_rev(j)] = X[t][j]
    dec = be.decrypt(ct_perm)
    err_perm = float(np.max(np.abs(np.array([dec[bitrev(j)] for j in range(hidden)]) - X[t])))
    print(f"  permute_in err: {err_perm:.3e}")

    # Step 2: bring to StoC level + StoC.
    # StoC needs depth = 30 - StoC_level. Already at depth 1 after gather.
    target_lvl = be._ops.slot_to_coeff_level()
    target_depth = 30 - target_lvl
    print(f"  StoC target_depth = {target_depth}")
    zero_ct = be.encrypt([0.0] * be._num_slots)
    while be._ops.depth(ct_perm) < target_depth:
        be._ops.mod_drop_inplace_ct(ct_perm)
    while be._ops.depth(zero_ct) < target_depth:
        be._ops.mod_drop_inplace_ct(zero_ct)
    t0 = time.time()
    ct_coeff = be._ops.slot_to_coeff(ct_perm, zero_ct, be._gk)
    t_stoc = time.time() - t0
    print(f"  stoc:        {t_stoc*1000:.1f}ms  depth_after={be._ops.depth(ct_coeff)}")

    # Sanity: decrypt coefficients, expect X[t] at coeffs [0, hidden)
    coeffs = be.decrypt_coeff(ct_coeff)
    err_coeff = float(np.max(np.abs(np.array(coeffs[:hidden]) - X[t])))
    print(f"  coeff err vs X[{t}]: {err_coeff:.3e}")

    if err_coeff > 1e-3:
        print("  ❌ stop: bit-rev permute did NOT produce contiguous coeffs after StoC")
        # Print where x actually lands
        # Find max-abs coefficients
        abs_c = np.abs(coeffs)
        top = np.argsort(abs_c)[-10:][::-1]
        print(f"  top-10 |coeff|: indices={top.tolist()}, values={abs_c[top].tolist()}")
        return

    # ── Bootstrap ct_coeff back to top of chain so nexus_linear has depth budget ──
    # nexus_linear needs depth=0 input. After StoC we're way down.
    print("  bootstrapping coeff ct...")
    # NOTE: bootstrap operates on slot-encoded inputs typically. May not work on coeff.
    # Try anyway.
    try:
        t0 = time.time()
        ct_boot = be.bootstrap(ct_coeff)
        t_boot = time.time() - t0
        print(f"  bootstrap:   {t_boot*1000:.1f}ms  depth_after={be._ops.depth(ct_boot)}")
        # Sanity-check after bootstrap
        bc = be.decrypt_coeff(ct_boot)
        err_b = float(np.max(np.abs(np.array(bc[:hidden]) - X[t])))
        print(f"  post-boot coeff err: {err_b:.3e}")
    except Exception as e:
        print(f"  bootstrap failed: {e}")
        ct_boot = ct_coeff


if __name__ == "__main__":
    main()
