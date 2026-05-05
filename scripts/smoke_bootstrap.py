"""Phase 8j smoke: verify HEonGPU bootstrap works at N=2^17 + measure cost.

Critical gate before L=128 plan: how fast is one bootstrap, what depth
do we recover, and does precision survive end-to-end on a ct that has
been pushed deep into the chain?
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # Match HEonGPU's working example exactly first.
    N = 1 << 18
    log(f"Init backend N={N} (scaling official bootstrap example)...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be._num_slots}")

    log("configure_bootstrapping...")
    t = time.time()
    be.configure_bootstrapping(
        CtoS_piece=3, StoC_piece=3, taylor_number=11, less_key_mode=True,
        slim=True, include_pow2_shifts=False,
    )
    log(f"  bootstrap params + galois keys ready in {time.time()-t:.1f}s")

    # Build a test vector with values that survive a long compute chain.
    rng = np.random.default_rng(1)
    test_vec = (rng.standard_normal(be._num_slots) * 0.1).tolist()

    log("Encrypt + push deep (mod_drop to slim depth)...")
    ct = be.encrypt(test_vec)
    init_depth = be._ops.depth(ct)
    log(f"  initial depth={init_depth}")

    # For slim bootstrap depth target = max_depth - StoC_piece (here 30-3=27).
    target_depth = be._max_depth - 3
    while be._ops.depth(ct) < target_depth:
        be._ops.mod_drop_inplace_ct(ct)
    log(f"  after mod-drop: depth={be._ops.depth(ct)} (slim target {target_depth})")

    # Decrypt before bootstrap to capture pre-state.
    pre = np.array(be.decrypt(ct))
    log(f"  pre-bootstrap recovery err: {np.max(np.abs(pre - test_vec)):.3e}")

    log("BOOTSTRAP...")
    t = time.time()
    ct_fresh = be.bootstrap(ct)
    bts_dt = time.time() - t
    log(f"  bootstrap took {bts_dt*1000:.0f} ms")
    log(f"  fresh depth={be._ops.depth(ct_fresh)} (recovered {be._max_depth - be._ops.depth(ct_fresh)} levels)")

    post = np.array(be.decrypt(ct_fresh))
    err = np.max(np.abs(post - test_vec))
    log(f"  post-bootstrap recovery err: {err:.3e}")
    log(f"  L2 err: {np.linalg.norm(post - test_vec):.3e}")

    print()
    print("=" * 70)
    print(f"BOOTSTRAP SMOKE @ N={N}")
    print(f"  cost            : {bts_dt*1000:.0f} ms")
    print(f"  depth recovered : {be._max_depth - be._ops.depth(ct_fresh)} levels")
    print(f"  precision       : max_err {err:.3e}")
    print("=" * 70)
    if err < 1e-2:
        print("PASS — bootstrap usable")
    else:
        print(f"FAIL — bootstrap precision {err:.3e} too low")


if __name__ == "__main__":
    main()
