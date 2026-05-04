"""Phase 7b-BSGS smoke test: validate BSGS path on a medium-sized matmul."""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor
from fhe_thesis.encryption.ops_matrix import enc_linear_matrix


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # n=256 → b1=b2=16. Forces BSGS (threshold=64).
    hidden_in, hidden_out, seq_len = 200, 256, 4
    block = 256

    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 8,        # need at least 4 levels
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready. max_depth={be._max_depth}")

    rng = np.random.default_rng(0)
    W = rng.standard_normal((hidden_out, hidden_in)) * 0.05
    b = rng.standard_normal(hidden_out) * 0.05
    x = rng.standard_normal((seq_len, hidden_in)) * 0.1
    expected = x @ W.T + b

    log(f"Encrypt input (block={block}, seq={seq_len}, hidden_in={hidden_in})...")
    ct_x = MatrixPackedTensor.encrypt(be, x, block=block)
    log(f"  cts={len(ct_x.cts)}  B={ct_x.tokens_per_ct}")

    log("Linear via enc_linear_matrix (BSGS path: n=256, b1=b2=16)...")
    t0 = time.time()
    h = enc_linear_matrix(be, ct_x, W, bias=b)
    dt = time.time() - t0
    log(f"  wall={dt*1000:.1f}ms  result depth={be._ops.depth(h.cts[0])}")

    out = h.decrypt(be)
    err = np.max(np.abs(out - expected))
    rel = err / max(1e-12, np.max(np.abs(expected)))
    log(f"  max abs err = {err:.3e}   rel = {rel:.3e}")
    log(f"  expected[0,:6] = {expected[0,:6]}")
    log(f"  got[0,:6]      = {out[0,:6]}")

    ok = err < 1e-3
    print()
    print("PASS" if ok else "FAIL", f"(err={err:.3e})")


if __name__ == "__main__":
    main()
