"""Debug: verify 1 round of decompress matches the expected math."""
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import required_galois_elts, enc_compress

N_LOG = 12
N = 1 << N_LOG

be = HEonGPUBackend(
    poly_modulus_degree=N,
    q_prime_bits=(60, 50, 50, 50),
    p_prime_bits=(60, 60, 60),
    scale_bits=50,
    sec_none=True,
)

elts = required_galois_elts(N)
print(f"N={N}, elts={elts[:3]}...")
gk = be._kg.generate_galois_key_elts(be._ctx, be._sk, elts)

rng = np.random.default_rng(2)
values = rng.normal(size=N).astype(np.float64) * 0.5

# Test 1: roundtrip encode_coeff/decode_coeff
pt0 = be._encoder.encode_coeff(be._ctx, list(values), be._scale)
ct0 = be._encryptor.encrypt(be._ctx, pt0)
dec0 = be._decryptor.decrypt(be._ctx, ct0)
rec0 = np.array(be._encoder.decode_coeff(dec0)[:N])
print(f"coeff roundtrip err: {np.max(np.abs(rec0 - values)):.2e}")

# Test 2: apply ψ_{N+1}: maps x to x^{N+1} = -x. So a(x) -> a(-x).
# Polynomial coefficients: c_j -> (-1)^j c_j.
ct_g = be._ops.apply_galois_elt(ct0, gk, N + 1)
dec_g = be._decryptor.decrypt(be._ctx, ct_g)
rec_g = np.array(be._encoder.decode_coeff(dec_g)[:N])
expected_g = values * np.array([(-1)**j for j in range(N)])
print(f"ψ_{{N+1}} coeffs err: {np.max(np.abs(rec_g - expected_g)):.2e}")
print(f"  sample rec_g[:6] = {rec_g[:6].round(4)}")
print(f"  sample exp[:6]   = {expected_g[:6].round(4)}")

# Test 3: t + ψ(t) should have only even coefficients (doubled), odd zeroed.
import fhe_thesis.encryption.heongpu_bindings._heongpu as hg
ct_sum = be._clone(ct0)
be._ops.add_inplace(ct_sum, ct_g)
dec_s = be._decryptor.decrypt(be._ctx, ct_sum)
rec_s = np.array(be._encoder.decode_coeff(dec_s)[:N])
expected_s = values + expected_g  # 2*values at even, 0 at odd
print(f"t + ψ(t) err: {np.max(np.abs(rec_s - expected_s)):.2e}")
print(f"  rec_s[:6] = {rec_s[:6].round(4)}")
print(f"  exp[:6]   = {expected_s[:6].round(4)}")

# Test 4: multiply_power_of_x by 2N-1 = -1 maps c_j to c_{j+1} for j<N-1, and -c_0 at position N-1.
ct_shift = be._clone(ct0)
be._ops.multiply_power_of_x_inplace(ct_shift, 2*N - 1)
dec_sh = be._decryptor.decrypt(be._ctx, ct_shift)
rec_sh = np.array(be._encoder.decode_coeff(dec_sh)[:N])
# Expected: shift down by 1, with negation at wrap (since multiplying by x^{-1} = -x^{N-1}).
# Coefficient j of (t * x^{2N-1}): t has coeff c_j at position j. x^{2N-1} = x^{-1} (under x^N=-1, mod x^{2N}-1?)
# Actually in negacyclic: x^N = -1, so x^{2N} = 1. x^{2N-1} = x^{-1} = -x^{N-1}.
# t * x^{N-1}: coeff at (j + N - 1) mod 2N. For j=1: N. So coefficient N+? wraps with sign flip.
# For coeff j of result: it's input coefficient (j - (2N-1)) mod 2N = (j+1) mod 2N, with sign flip if wrapped odd number of N.
# (j+1) for j<N-1 stays in [1, N-1]: input coeff j+1, no wrap. Then negation factor from "* -1" = 1.
# Wait — multiplying by x^{2N-1} = -x^{N-1}: result coefficient at position k is input coeff at (k - (N-1)) mod N, times sign.
# Hmm, let me just test the per-coeff result.
print(f"  shift x^(2N-1):")
print(f"  rec_sh[:4] = {rec_sh[:4].round(4)}")
print(f"  rec_sh[N-1:N] = {rec_sh[N-2:N].round(4)}")
print(f"  values[:4] = {values[:4].round(4)}")
print(f"  values[N-1] = {values[N-1].round(4)}")
