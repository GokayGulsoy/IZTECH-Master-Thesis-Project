"""Debug matmul with tiny dims, print actual vs expected."""
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend

bk = HEonGPUBackend(poly_modulus_degree=2**14, sec_none=True)
n = bk.num_slots

# Tiny: 4x4
in_dim, out_dim = 4, 4
W = np.eye(4, dtype=np.float64)  # identity → expect output = input
xv = np.array([1.0, 2.0, 3.0, 4.0])
padded = np.zeros(n)
padded[:in_dim] = xv
ct = bk.encrypt(padded.tolist())

mm = bk.matmul_plain(ct, W.tolist(), None)
got = np.asarray(bk.decrypt(mm))[:out_dim]
print("identity W, x =", xv)
print("got      =", got)
print("expected =", W @ xv)
print("err      =", np.max(np.abs(got - W @ xv)))

# Now W = ones
W2 = np.ones((4, 4), dtype=np.float64)
mm2 = bk.matmul_plain(ct, W2.tolist(), None)
got2 = np.asarray(bk.decrypt(mm2))[:out_dim]
print("\nones W, expected = [10,10,10,10]")
print("got =", got2)
print("err =", np.max(np.abs(got2 - W2 @ xv)))
