"""Phase 8j-2: verify slim bootstrap + BSGS rotation keys coexist at N=2^17.

Earlier finding: chain depth=30 gives 9 useful levels recovered per bootstrap.
With chain=22 we only get 2, which is too few for a BERT layer.

Strategy: use depth=30 chain + register ONLY the rotation shifts the BERT
pipeline actually needs (BSGS shifts for hidden=768/intermediate=3072
matvec), not the full ±2^k set.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import prepare_colmajor_keys


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    N = 1 << 17
    log(f"Init backend N={N} chain=24...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 24,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be._num_slots}")

    log("configure_bootstrapping (slim, NO blanket pow2 shifts)...")
    t = time.time()
    be.configure_bootstrapping(
        CtoS_piece=3, StoC_piece=3, taylor_number=11, less_key_mode=True,
        slim=True, include_pow2_shifts=False,
    )
    log(f"  ready in {time.time()-t:.1f}s — {len(be._registered_shifts)} bootstrap shifts")

    log("register BERT-specific col-major rotation keys (max_dim=3072 for FFN)...")
    t = time.time()
    n_new = prepare_colmajor_keys(be, L=128, max_dim=3072)
    log(f"  +{n_new} new shifts in {time.time()-t:.1f}s "
        f"-> total {len(be._registered_shifts)} shifts")

    rng = np.random.default_rng(1)
    test_vec = (rng.standard_normal(be._num_slots) * 0.1).tolist()

    ct = be.encrypt(test_vec)
    target = be._max_depth - 3
    while be._ops.depth(ct) < target:
        be._ops.mod_drop_inplace_ct(ct)
    log(f"  pushed to depth={be._ops.depth(ct)} (slim target={target})")

    log("BOOTSTRAP #1...")
    t = time.time()
    ct1 = be.bootstrap(ct)
    bts1 = time.time() - t
    fresh_depth1 = be._ops.depth(ct1)
    err1 = np.max(np.abs(np.array(be.decrypt(ct1)) - test_vec))
    log(f"  done in {bts1*1000:.0f} ms, fresh depth={fresh_depth1} err={err1:.3e}")
    usable_levels = be._max_depth - fresh_depth1
    log(f"  usable levels available after BTS: {usable_levels}")

    log("Test rotate by 28 (BSGS giant step shift)...")
    t = time.time()
    ct_rot = be.rotate(ct1, 28)
    rot_dt = time.time() - t
    rot_dec = np.array(be.decrypt(ct_rot))
    rot_err = np.max(np.abs(rot_dec - np.roll(test_vec, -28)))
    log(f"  rotate 28 took {rot_dt*1000:.0f} ms, err={rot_err:.3e}")

    log("Squarings to consume levels, then BOOTSTRAP #2 (chain stability)...")
    work = ct1
    n_sq = 0
    while be._ops.depth(work) < target:
        if be._ops.depth(work) <= target - 1:
            work = be.mul_relin(work, work)
            n_sq += 1
        else:
            be._ops.mod_drop_inplace_ct(work)
    log(f"  did {n_sq} squarings, depth={be._ops.depth(work)}")

    t = time.time()
    ct2 = be.bootstrap(work)
    bts2 = time.time() - t
    log(f"  BTS2 done in {bts2*1000:.0f} ms, fresh depth={be._ops.depth(ct2)}")

    print()
    print("=" * 70)
    print(f"BOOTSTRAP+KEYS @ N={N} chain={be._max_depth+1}")
    print(f"  total Galois shifts  : {len(be._registered_shifts)}")
    print(f"  bootstrap cost        : BTS1={bts1*1000:.0f}ms  BTS2={bts2*1000:.0f}ms")
    print(f"  usable levels per BTS : {usable_levels}")
    print(f"  precision (BTS1)      : {err1:.3e}")
    print(f"  rotation err          : {rot_err:.3e}")
    print("=" * 70)
    if err1 < 1e-2 and rot_err < 1e-2 and usable_levels >= 6:
        print("PASS — bootstrap + BSGS keys coexist at N=2^17 with usable headroom")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
