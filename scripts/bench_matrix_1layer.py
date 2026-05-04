"""1-layer BERT-base matrix-packed bench with live per-stage timing.

Goal: get a real per-stage cost breakdown for the matrix pipeline FAST,
so we can see the actual bottleneck (linears vs attention vs LN vs GELU)
on real shapes.
"""
import sys
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.protocol import (
    LayerWeights, encrypt_layer_matrix,
)
from fhe_thesis.encryption.coefficients import PolyCoeffs
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    hidden, num_heads, seq_len = 768, 12, 8

    log(f"BERT-base 1-layer, hidden={hidden}, heads={num_heads}, seq={seq_len}")
    log("Init backend...")
    t = time.time()
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready in {time.time()-t:.1f}s")

    log("Build weights...")
    rng = np.random.default_rng(0)
    layer = LayerWeights(
        Wq=rng.standard_normal((hidden, hidden)) * 0.02, bq=np.zeros(hidden),
        Wk=rng.standard_normal((hidden, hidden)) * 0.02, bk=np.zeros(hidden),
        Wv=rng.standard_normal((hidden, hidden)) * 0.02, bv=np.zeros(hidden),
        Wo=rng.standard_normal((hidden, hidden)) * 0.02, bo=np.zeros(hidden),
        W1=rng.standard_normal((4*hidden, hidden)) * 0.02, b1=np.zeros(4*hidden),
        W2=rng.standard_normal((hidden, 4*hidden)) * 0.02, b2=np.zeros(hidden),
        ln1_gamma=np.ones(hidden), ln1_beta=np.zeros(hidden),
        ln2_gamma=np.ones(hidden), ln2_beta=np.zeros(hidden),
    )
    coeffs = {
        "GELU":    PolyCoeffs(power_coeffs=np.array([0.5,0.5,0,0,0,0]), interval=(-4.,4.), degree=5),
        "Softmax": PolyCoeffs(power_coeffs=np.array([0.5,0,0.5,0,0,0]), interval=(-8.,8.), degree=5),
        "LN":      PolyCoeffs(power_coeffs=np.array([1.,-0.5,0.375,-0.3125,0.273,-0.246]), interval=(0.01,4.), degree=5),
    }

    x = rng.standard_normal((seq_len, hidden)) * 0.1
    block = next_pow2(4 * hidden)  # FFN expansion
    log(f"Encrypt input (block={block})...")
    t = time.time()
    ct_x = MatrixPackedTensor.encrypt(be, x, block=block)
    log(f"  encrypted in {time.time()-t:.2f}s  (cts={len(ct_x.cts)}, B={ct_x.tokens_per_ct})")

    log("Run 1 encoder layer...")
    t = time.time()
    h, timings = encrypt_layer_matrix(be, ct_x, layer, coeffs, num_heads)
    wall = time.time() - t
    log(f"  LAYER WALL: {wall:.2f}s")
    print()
    print("Per-stage:")
    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s} {v:7.3f}s")
    proj_total_layers = wall * 12
    print(f"\nProjected 12-layer wall: {proj_total_layers:.1f}s = {proj_total_layers/60:.1f}min")


if __name__ == "__main__":
    main()
