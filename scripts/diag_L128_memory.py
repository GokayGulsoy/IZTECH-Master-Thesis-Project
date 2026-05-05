"""Phase 8h-diag: quantify L=128 GPU memory usage to decide
HEonGPU-fix vs Phantom-port.

Steps:
  1. boot N=2^17 with FULL pow-of-2 key set (current default) -> measure mem
  2. boot N=2^17 with EMPTY initial key set, then add only the shifts we need
     for one BERT-base layer (col-major linear BSGS + LN bcast + attn)
  3. report: how much mem do the eval keys cost, how much do cts cost
"""
import time, gc, sys
import subprocess

def nvsmi():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free",
         "--format=csv,noheader,nounits"]
    ).decode().strip().split(",")
    return int(out[0]), int(out[1])  # MiB

def log(m): print(f"[{time.strftime('%H:%M:%S')}] [used={nvsmi()[0]}MiB] {m}", flush=True)

def main():
    L = 128
    hidden = 768
    head_dim = 64
    N = 1 << 17  # n_slots = 65536; need L*hidden = 98304? -> overflows.
    # Actually L=128 hidden=768 = 98304 slots > 65536. Need N=2^18.
    # But let's first try N=2^17 with hidden_padded that fits per slab.
    # For diagnostic just use N=2^17 to bound the key-cost.

    log("=== STEP 1: default backend N=2^17 (full pow-of-2 keys) ===")
    from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
    t0 = time.time()
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 14,  # smaller chain to isolate key cost
        p_prime_bits=(60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  backend up in {time.time()-t0:.1f}s, max_depth={be._max_depth}")
    log(f"  initial registered shifts: {len(be._registered_shifts)}")

    # Add the actual rotations we need for one BERT-base attention layer.
    from fhe_thesis.encryption.ops_attention_nexus import prepare_colmajor_keys
    log("Adding L=128 col-major key set (max_dim=2048)...")
    t = time.time()
    n_added = prepare_colmajor_keys(be, L=L, max_dim=2048)
    log(f"  +{n_added} keys in {time.time()-t:.1f}s; "
        f"total registered = {len(be._registered_shifts)}")

    # Now allocate one ciphertext at full ring size to measure ct cost.
    log("Encrypt one ct...")
    ct = be.encrypt([0.1] * be._num_slots)
    log(f"  one ct allocated")

    log("=== DONE ===")
    print(f"\nFinal memory snapshot: used={nvsmi()[0]}MiB, free={nvsmi()[1]}MiB")
    print(f"Galois key count: {len(be._registered_shifts)}")
    print(f"Chain depth: {be._max_depth}")


if __name__ == "__main__":
    main()
