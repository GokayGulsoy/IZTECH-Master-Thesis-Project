"""Minimal isolation test for per_block_rotate_left on HEonGPU."""
from __future__ import annotations
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_matrix import per_block_rotate_left


def main() -> int:
    be = HEonGPUBackend(
        poly_modulus_degree=32768,
        q_prime_bits=[60] + [40] * 18,
        p_prime_bits=[60],
        scale_bits=40,
    )
    n = be.capabilities.n_slots
    block = 64
    shift = 32

    # Construct a vector where block 0 holds 1..32 in positions [0:32], zeros in [32:64].
    # All other blocks zero. Easy to read after rotation.
    v = [0.0] * n
    for j in range(32):
        v[j] = float(j + 1)        # 1..32 in positions [0:32]
    # Block 1 (slots [64:128]): put 100..131 to detect cross-block leakage
    for j in range(32):
        v[64 + j] = float(100 + j)

    ct = be.encrypt(v)
    out = np.asarray(be.decrypt(ct))
    print(f"input  block0 [0:5]={out[:5]} ... [30:35]={out[30:35]} ... [60:68]={out[60:68]}")
    print(f"input  block1 [0:5]={out[64:69]}")

    rot = per_block_rotate_left(be, ct, shift, block=block, num_slots=n)
    out = np.asarray(be.decrypt(rot))
    print(f"\nrotate(block_left {shift}) within block of {block}:")
    print(f"out block0 [0:5]   = {out[:5]}    expected~[0,0,0,0,0]")
    print(f"out block0 [30:35] = {out[30:35]}  expected~[0,0,1,2,3]")
    print(f"out block0 [60:68] = {out[60:68]} expected~[29,30,31,32,0,...]")
    print(f"out block1 [0:5]   = {out[64:69]}  expected~[0,0,0,0,0]")
    print(f"out block1 [30:35] = {out[64+30:64+35]}  expected~[0,0,100,101,102]")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
