"""Diagnose: trace what enc_linear_matrix does step-by-step on HEonGPU.

Reproduces the failing 8x32->16 case but inspects intermediate
ciphertexts after each kernel stage so we can spot where the math
diverges from the expected plaintext result.
"""
from __future__ import annotations
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption.ops_matrix import (
    _block_masks,
    _replicate_in_block,
    per_block_rotate_left,
)


def main() -> int:
    seq_len, in_dim, out_dim = 8, 32, 16
    block = 64

    rng = np.random.default_rng(0)
    X = rng.standard_normal((seq_len, in_dim)).astype(np.float64) * 0.5
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float64) * 0.1

    be = HEonGPUBackend(
        poly_modulus_degree=32768,
        q_prime_bits=[60] + [40] * 18,
        p_prime_bits=[60],
        scale_bits=40,
    )
    print(f"slots={be.capabilities.n_slots}  scale={be.scale}")

    mpt = MatrixPackedTensor.encrypt(be, X, block=block)
    ct0 = mpt.cts[0]
    raw = np.asarray(be.decrypt(ct0))
    # Verify packing layout
    print(f"\nraw[0:5]  = {raw[:5]}")
    print(f"raw[64:69]= {raw[64:69]}  (expected token 1)")
    print(f"X[0,:5]   = {X[0,:5]}")
    print(f"X[1,:5]   = {X[1,:5]}")
    print(f"max packing err = {np.max(np.abs(raw[:32] - X[0])):.4g}")

    # Apply replicate
    rep = _replicate_in_block(be, ct0, in_dim=in_dim, block=block,
                              num_slots=be.capabilities.n_slots)
    raw_rep = np.asarray(be.decrypt(rep))
    # After replicate, block 0 should hold X[0] cyclically replicated to length 64
    expected_rep_block0 = np.concatenate([X[0], X[0]])  # 32+32=64
    expected_rep_block1 = np.concatenate([X[1], X[1]])
    print(f"\nrep block 0 [0:5]   = {raw_rep[:5]}")
    print(f"rep block 0 [32:37] = {raw_rep[32:37]}  (expected = X[0,:5])")
    print(f"max replicate err   = {np.max(np.abs(raw_rep[:64] - expected_rep_block0)):.4g}")
    print(f"max replicate err b1= {np.max(np.abs(raw_rep[64:128] - expected_rep_block1)):.4g}")

    # Apply per_block_rotate_left by 1 to verify
    rot = per_block_rotate_left(be, rep, 1, block=block,
                                num_slots=be.capabilities.n_slots)
    raw_rot = np.asarray(be.decrypt(rot))
    expected_rot_block0 = np.roll(expected_rep_block0, -1)
    print(f"\nrot1 block 0 [0:5]  = {raw_rot[:5]}")
    print(f"expected            = {expected_rot_block0[:5]}")
    print(f"max rot1 err        = {np.max(np.abs(raw_rot[:64] - expected_rot_block0)):.4g}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
