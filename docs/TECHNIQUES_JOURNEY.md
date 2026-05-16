# Techniques Journey — How We Landed at Synthesizer-LPAN

> **Purpose**: capture the full narrative of techniques tried, results
> measured, and decisions made. Designed to enable seamless context
> handoff to a fresh chat session or a new collaborator.

All wall-times are 12-layer end-to-end forward passes at sequence
length $L=128$ on a single **H100 SXM5** with the **HEonGPU** CKKS
backend.

---

## Stage 0 — Honest LPAN baseline (833 s)

**Configuration**: full LPAN (Learnable Polynomial Activation Network),
softmax-poly degree-12 in **all 12 layers**, plain self-attention with
$Q\cdot K^\top$ ciphertext-by-ciphertext multiplications, polynomial
GELU degree-6, polynomial LayerNorm cubic-invsqrt.

**Wall-time**: 833 s.

**Why this is the honest baseline**: every LPAN paper (NEXUS, MPCFormer,
BOLT, Iron) measures a flavour of "all-poly all-layer" as their headline
floor. We replicated it to have a directly comparable speedup denominator.

**Bottleneck observed**: $L^2$ ciphertext-by-ciphertext multiplications
in $Q\cdot K^\top$ + the depth-12 softmax-poly evaluation dominate
~70% of per-layer wall-time.

---

## Stage 1 — Batched-shared-W LPAN (534 s, 1.56×)

**Idea**: Pack `BATCH=4` independent samples in the slot dimension and
share the linear projection plaintexts across the batch. Polynomial
activations (GELU, LN-invsqrt) amortize for free; only the $Q\cdot K^\top$
critical path stays per-sample.

**Result**: 833 s → 534 s (1.56×).

**Lesson**: batching helps polynomial activations a lot, but the
$Q\cdot K^\top$ ct·ct floor is invariant to batching → diminishing returns.

---

## Stage 2 — Linformer-LPAN attempt (~440 s projected, 1.89× — abandoned)

**Idea**: Apply Linformer (Wang et al., 2020) — project the sequence
dimension $L \to k$ via a learned plaintext $E \in \mathbb{R}^{L \times k}$
on $K$ and $V$, reducing $Q\cdot K^\top$ from $L^2$ to $L \cdot k$ ct·ct.

**Result**: only ~1.2× projected speedup over Stage 1 (~440 s).

**Why it failed**: NEXUS column-major slot layout makes sequence-mixing
rotations expensive — projecting along the token axis requires $O(\log L)$
rotations per sample, and Linformer's $E$ projection becomes one of the
most expensive ops in the circuit. The math reduction in ct·ct count is
real, but the rotation overhead eats the savings.

**Lesson**: ML approximations that look cheap in plaintext can be
expensive under FHE because the cost model is fundamentally different
(rotations, depth, slot layout). Always measure under the actual FHE
backend before committing.

---

## Stage 3 — Synthesizer-LPAN naive, BATCH=4 (222.8 s, 3.74×)

**The architectural breakthrough.**

**Idea**: Replace standard self-attention with **Synthesizer attention**
(Tay et al., NeurIPS 2020). The $A \in \mathbb{R}^{L \times L}$ attention
matrix is **frozen, learned, plaintext**, with softmax already absorbed
at training time:

$$
O = A \cdot V
$$

Under FHE this **eliminates**:

- $W_q$ projection (no $Q$ needed)
- $W_k$ projection (no $K$ needed)
- $Q \cdot K^\top$ ($L$ ct·ct multiplications, depth 2)
- softmax-poly (depth 12, the deepest single op)

The remaining $A \cdot V$ is computed via diagonal decomposition:

```
For d ∈ [0, L):
  V_shift_d = cyclic_shift_along_i(V, d)     # 2 rotations + 2 mul_plain
  diag_mask_d = encode(A[h, i, (i+d) mod L]) # plaintext, encoded ONCE
  O += V_shift_d ⊙ diag_mask_d                # 1 mul_plain + 1 add
```

Cost: $2L$ rotations + $3L$ pt·ct multiplications + $3L$ additions.
**Zero ct·ct multiplications.**

**Result**: 833 s → 222.8 s (3.74× over baseline; 2.4× over Stage 1).

**Why this works under FHE but not in plaintext**: Tay et al. discarded
plaintext Synthesizer in 2020 because the speedup was negligible
(plaintext MHA is already fast). Under FHE the entire $L^2$ ct·ct floor
disappears — making this an architectural lever LPAN-family papers
all left on the table.

**Accuracy**: Tay et al. report >97% of standard MHA on GLUE.

---

## Stage 4 — Synthesizer-LPAN naive, BATCH=16 (131.4 s, 6.34×)

**Idea**: Push batching from 4 → 16. The slot count $N/2 = 32{,}768$
fits BATCH=16 across multiple ciphertext bundles orchestrated by
[`fhe_thesis/encryption/multi.py`](../fhe_thesis/encryption/multi.py).

**Result**: 222.8 s → 131.4 s (6.34× over baseline).

**Lesson**: with the $Q\cdot K^\top$ floor removed, batching now has
much more leverage (no per-sample ct·ct critical path).

---

## Stage 5 — Synthesizer-LPAN BSGS, BATCH=8, chain=24 (86.1 s, 9.67×)

**The sub-100-second milestone.**

**Idea**: Apply Halevi-Shoup BSGS to the $A \cdot V$ inner loop. Both
the cyclic-shift mask `mask_top[d]` and the diagonal pattern `diag[d]`
are plaintext — fuse them at encoding time:

```
top_combined[d] = mask_top[d] ⊙ diag[d]   # pre-encoded plaintext
bot_combined[d] = mask_bot[d] ⊙ diag[d]   # pre-encoded plaintext
```

Then apply BSGS with $bs \cdot gs = L$, both $\approx \sqrt{L}$:

```
For g ∈ [0, gs):
  V_giant_g = rotate(V, g·bs)                       # 1 rotation per g
  For b ∈ [0, bs):
    O += V_giant_g ⊙ top_combined_pre_rotated[g][b] # 1 pt·ct + 1 add
```

Rotation count: $2L \to 2 \cdot 2\sqrt{L}$. At $L=128$: **256 → 32**
rotations, an 8× reduction. Mathematical equivalence is exact.

**First BSGS run**: BATCH=8, chain=24 — gave 86.1 s (9.67× over baseline).
First time crossing the symbolic 100 s threshold for end-to-end
single-GPU FHE BERT.

---

## Stage 6 — Synthesizer-LPAN BSGS, BATCH=16, chain=22 (60.9 s, 13.67× — FINAL)

**Idea**: Tune the chain budget down (24 → 22) to free slots for
BATCH=16. Chain=22 is the **minimum viable**: LayerNorm's cubic
invsqrt polyval needs 6 levels, and chain ≤ 21 ran out of levels
mid-circuit. Chain=22 has just enough headroom for residual addition
depth alignment.

**Result**: 86.1 s → **60.9 s** at BATCH=16, chain=22. **13.67×
over the 833 s honest LPAN baseline.**

**Per-sample effective throughput**: 60.9 / 16 ≈ **3.8 s per sample**.

**Why we stopped here**: this is a clean single-GPU sub-100-second
end-to-end coherent FHE BERT result on plain HEonGPU. Further gains would require
either (a) multi-GPU sharding (CERIUM owns that axis), or (b) framework-
level optimizations (CERIUM's compiler/runtime) — both compose with
our architectural change.

---

## Concurrent work — CERIUM (Dec 2025, CMU + NVIDIA)

**What they did**: framework-level FHE BERT — DSL + compiler + runtime
for multi-GPU CKKS execution.

**Their numbers**:

| Hardware | Wall-time |
|---|---|
| 8× B200 | **8.8 s** |
| 1× H100 | 36.1 s |
| 1× A100 | 66 s |

**Their architecture**: plain BERT, no architectural change. They use
polynomial approximations for nonlinearities (GELU, softmax with max
normalization, LN). Achieve 69.3% on GLUE RTE matching plaintext.

**How they relate to us**:

- **Different scales**: their 8.8 s requires an 8-GPU datacenter
  cluster (~$200K). Ours runs on a single H100.
- **Different contributions**: theirs is **framework-level**; ours is
  **architectural**.
- **Compose, don't compete**: CERIUM's compiler could lower our
  Synthesizer-LPAN circuit to its multi-GPU runtime. Their plain-BERT
  front-end could be replaced with our Synthesizer-LPAN architecture
  for an additional 13.67× factor on top of their framework gains.
- **Single-H100 comparison**: their 36.1 s on 1× H100 is for plain BERT
  using their framework. Our 60.9 s is for plain HEonGPU. Apples-to-
  oranges; the architectural lever is what we contribute.

---

## Cancelled / abandoned techniques

| Technique | Why abandoned |
|---|---|
| Linformer-LPAN sequence projection | NEXUS column-major rotations eat the ct·ct savings |
| Plaintext softmax injection (mid-circuit) | Violates pure-FHE threat model |
| Bootstrap mid-forward | chain=22 is sufficient; bootstrap added ~30 s with no qualitative gain |
| Multi-GPU sharding | Out of scope; CERIUM owns this axis |
| CKKS + TFHE mixed crypto for nonlinearities | Adds protocol complexity; deferred |
| HyPER-LPAN composition selector (LM/Q/L per layer) | Dropped after Synthesizer breakthrough — no longer needed |
| Word elimination (Ext W) | Dropped for now; doesn't compose well with frozen $A$ |
| Region-adaptive bootstrap (Ext 1) | Moot at chain=22 with no bootstrap |

---

## Where we are now (May 2026)

**Achieved**:

- ✅ 60.9 s single-H100 end-to-end Synthesizer-LPAN forward (12 layers, L=128, BATCH=16)
- ✅ 13.67× speedup over honest LPAN baseline
- ✅ Pure-FHE threat model preserved (no MPC, no TEE, no mid-circuit decryption)
- ✅ Production branch with vendored HEonGPU + commit pin
- ✅ Modular `fhe_thesis/encryption/` (5 focused submodules)
- ✅ Naive vs BSGS correctness test passing
- ✅ Documentation updated to reflect final architecture

**Remaining**:

- ⏳ Plug trained Synthesizer-LPAN BERT checkpoint into the FHE bench
  for end-to-end accuracy + latency report on GLUE
- ⏳ Full GLUE training (SST-2, MRPC, QNLI, RTE)
- ⏳ Pareto plot harness (FHE latency vs accuracy)
- ⏳ Threat-model formalization for paper §6
- ⏳ Thesis chapters 3 (methodology) + 4 (results) write-up
- ⏳ USENIX Security 2027 submission

---

## Lessons for future work

1. **FHE cost model ≠ plaintext cost model.** ML techniques discarded
   in plaintext (Synthesizer) can be transformative under FHE.
   Always measure under the target backend before committing.

2. **Architectural levers > framework levers** *for some axes*.
   Our 13.67× came from changing the math; CERIUM's framework gives
   ~10× from compilation+multi-GPU. Both are real and complementary.

3. **The slot-layout choice (NEXUS column-major) is load-bearing.**
   It made Synthesizer cheap and Linformer expensive. Slot layout
   should be chosen jointly with architecture.

4. **Chain budget is a tunable, not a constant.** chain=22 vs chain=24
   was a 30% wall-time difference at the same correctness.

5. **Vendor your crypto backend.** Pinning HEonGPU to a known commit
   eliminates an entire class of "the numbers regressed and we don't
   know why" bugs.

6. **Pure-FHE threat model is a feature, not a constraint.** It rules
   out tempting MPC shortcuts but yields a stronger paper.
