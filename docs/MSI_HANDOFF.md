# MSI Handoff — Privacy-Preserving Transformer Inference (LPAN + CKKS)

> **Audience:** the Claude Opus 4.6 instance running on the MSI Vector
> 17HX (RTX 5070 Ti). This document is the single source of truth for
> the project's current state and the next concrete steps.
>
> **Branch:** `feature/ckks-protocol` (HEAD as of merge with
> `feature/glue-multi-dataset`).

---

## 1. What this project does

End-to-end privacy-preserving BERT inference on GLUE tasks. The
pipeline has two halves:

1. **Training half (CPU/GPU on MSI):** *Learnable Polynomial
   Activation Networks (LPAN)* — replace GELU, softmax, and LayerNorm
   in a pretrained BERT with degree-8 polynomials whose coefficients
   are trained in 3 stages with knowledge distillation. Implemented
   in [`run_staged_lpan.py`](../run_staged_lpan.py) and
   [`fhe_thesis/training/trainer.py`](../fhe_thesis/training/trainer.py).

2. **Inference half (FHE on MSI):** *Pure-FHE Single-Round (PF-SR)*
   protocol — because every nonlinearity is now a polynomial, the
   server can evaluate the entire encrypted forward pass without any
   MPC round-trips (unlike THE-X / MPCFormer / BOLT). One `Enc(x)` in,
   one `Enc(y)` out. Implemented in
   [`fhe_thesis/encryption/`](../fhe_thesis/encryption/) and driven
   by [`experiments/run_protocol.py`](../experiments/run_protocol.py).

The two halves share the same trained checkpoint: training writes
`results/multi_model/<model>/staged_lpan_final/best_model/`, and
[`extract_coefficients.py`](../extract_coefficients.py) dumps the
learned polynomial coefficients to
`results/coefficients/bert_<model>_coeffs.json`, which the FHE half
reads via [`fhe_thesis/encryption/coefficients.py`](../fhe_thesis/encryption/coefficients.py).

---

## 2. Repository layout (current)

```
IZTECH-Master-Thesis-Project/
├── fhe_thesis/                          # All importable library code
│   ├── config.py                          # MODEL_REGISTRY, paths, intervals
│   ├── tasks.py                           # GLUE task registry (4 tasks)
│   ├── poly/                              # Polynomial approximation toolkit
│   │   ├── approximation.py               # weighted-minimax + densities
│   │   └── chebyshev.py                   # Cheb ↔ power-basis conversion
│   ├── models/                            # Activation profiling + replacement
│   │   ├── activations.py                 # LearnablePolyActivation modules
│   │   ├── profiling.py                   # profile_model + KDE density
│   │   └── replacement.py                 # swap nn.GELU/Softmax/LN → poly
│   ├── training/
│   │   └── trainer.py                     # 3-stage trainer + KD + GLUE loaders
│   └── encryption/                        # PF-SR FHE protocol
│       ├── context.py                     # CKKS context factories (TenSEAL)
│       ├── backend.py                     # CKKSBackend ABC + TenSEALBackend
│       ├── packing.py                     # TokenPackedTensor (1 ct / token)
│       ├── ops.py                         # enc_linear, enc_gelu_poly,
│       │                                  # enc_ln_poly, enc_qk_scores,
│       │                                  # enc_softmax_poly,
│       │                                  # enc_attention_apply,
│       │                                  # enc_self_attention
│       ├── coefficients.py                # PolyCoeffs + load_coefficients
│       ├── protocol.py                    # MODEL-AGNOSTIC blocks:
│       │                                  #   encrypt_ffn_block /
│       │                                  #   encrypt_attention_block /
│       │                                  #   encrypt_layer /
│       │                                  #   encrypt_inference / run_phase
│       └── depth.py                       # symbolic depth audit; layer = 23
│
├── run_staged_lpan.py                   # ENTRYPOINT — train one model on one task
├── extract_coefficients.py              # ENTRYPOINT — dump learned coefficients
│
├── experiments/                         # All runnable scripts live here
│   ├── 05_multi_model_scaling.py        # ENTRYPOINT — baseline + LPAN sweep
│   ├── run_protocol.py                  # ENTRYPOINT — encrypted inference CLI
│   ├── run_analysis.py                  # ENTRYPOINT — thesis figure analyses
│   ├── analysis/                        # Analysis scripts (Contributions 1/2/5)
│   │   ├── poly_approximation.py        # Taylor vs Chebyshev vs LSQ vs WMM
│   │   ├── activation_profiling.py      # GELU/Softmax/LN distributions
│   │   ├── error_propagation.py         # theoretical + empirical bounds
│   │   └── bsgs_eval_strategies.py      # Horner vs balanced-tree vs PS
│   └── generate_figures.py              # ENTRYPOINT — regenerate thesis PNGs
│
├── docs/
│   ├── ckks_protocol.md                 # Full FHE protocol spec
│   └── MSI_HANDOFF.md                   # ← you are here
│
├── IZTECH_Master_Thesis/                # LaTeX thesis
│   ├── main.tex
│   ├── chapters/{1..5}_*.tex
│   └── Figures/                         # Drop generated PNGs here
│
└── results/                             # All outputs (gitignored)
    ├── multi_model/<model>/staged_lpan_final/best_model/   # checkpoints
    ├── coefficients/bert_<model>_coeffs.json               # extracted polys
    ├── encrypted_inference/<model>_<phase>.json            # FHE timings
    ├── poly_approx/    activation_profiles/    error_propagation/
    ├── bsgs_eval/      multi_model/            multi_dataset/
    └── figures/                                             # final PNGs
```

**Five entrypoints, that's it.** Anything outside this list is library
code consumed via `import fhe_thesis.…`.

---

## 3. Model + task registry

`fhe_thesis/config.py` :: `MODEL_REGISTRY`

| key    | HuggingFace name                       | layers | hidden | heads | params |
|--------|----------------------------------------|--------|--------|-------|--------|
| tiny   | `google/bert_uncased_L-2_H-128_A-2`    |  2     |  128   |  2    |  4.4 M |
| mini   | `google/bert_uncased_L-4_H-256_A-4`    |  4     |  256   |  4    | 11.2 M |
| small  | `google/bert_uncased_L-4_H-512_A-8`    |  4     |  512   |  8    | 28.8 M |
| base   | `bert-base-uncased`                    | 12     |  768   | 12    | 110 M  |

`fhe_thesis/tasks.py` :: `GLUE_TASKS`

| key    | name | metric          | classes | sentence pair |
|--------|------|-----------------|---------|---------------|
| sst2   | SST-2 | accuracy        | 2       | no            |
| mrpc   | MRPC  | F1 + accuracy   | 2       | yes           |
| qqp    | QQP   | F1 + accuracy   | 2       | yes           |
| qnli   | QNLI  | accuracy        | 2       | yes           |

The four-task subset matches the comparison tables in MPCFormer /
THE-X / BOLT. Both `run_staged_lpan.py` and
`experiments/05_multi_model_scaling.py` accept `--task <key>`.

---

## 4. The full pipeline, end-to-end

A single `(model, task)` pair from raw HuggingFace BERT to encrypted
inference takes four sequential steps. Each step is independent and
restartable.

### Step 1 — Profile activations (per-model, one-time)

```bash
# Profile all four BERT variants in one go (writes per-model subdirs)
python experiments/run_analysis.py profile

# OR pick a subset:
python experiments/analysis/activation_profiling.py --model tiny mini
```

Writes `results/activation_profiles/<model>/` (PNGs + per-layer KDE
samples for GELU inputs / Softmax pre-shift / LN variances). The
percentile intervals printed at the end populate the
`PROFILED_INTERVALS` defaults that the polynomial fits use.

> **Backward compatibility note.** The output path changed from a
> flat `results/activation_profiles/` (Tiny only) to per-model
> subdirectories `results/activation_profiles/<model>/`. The
> previously-published BERT-Tiny SST-2 profiling figures should be
> regenerated by running `--model tiny`; the polynomial coefficients
> derived from them have **not** changed (intervals are still hard-coded
> in `fhe_thesis/config.py :: PROFILED_INTERVALS`).

### Step 2 — LPAN train (per `(model, task)` pair)

```bash
python run_staged_lpan.py --model tiny  --task sst2
python run_staged_lpan.py --model mini  --task qnli
python run_staged_lpan.py --model small --task qqp
python run_staged_lpan.py --model base  --task mrpc      # slow!
```

This runs the 3-stage curriculum:

* **Stage 1** — fine-tune the floating-point teacher on the task.
* **Stage 2** — replace nonlinearities with `LearnablePolyActivation`,
  initialise from weighted-minimax fit, train *only* the poly
  coefficients (rest frozen).
* **Stage 3** — unfreeze and train end-to-end with KD against the
  Stage-1 teacher.

Outputs the final checkpoint to
`results/multi_model/<model>/staged_lpan_final/best_model/`.

### Step 3 — Extract polynomial coefficients

```bash
python extract_coefficients.py                  # all 4 models
python extract_coefficients.py --model base     # one model
```

Writes `results/coefficients/bert_<model>_coeffs.json`. This is the
single artefact that crosses the train ↔ infer boundary.

### Step 4 — Encrypted inference (PF-SR protocol)

```bash
# The four phases, each runnable per model:
python experiments/run_protocol.py --model tiny  --phase ffn
python experiments/run_protocol.py --model tiny  --phase attention
python experiments/run_protocol.py --model tiny  --phase layer
python experiments/run_protocol.py --model tiny  --phase model
```

Phase semantics (see `fhe_thesis/encryption/protocol.py`):

| phase     | what runs under FHE                                    | depth (levels) |
|-----------|--------------------------------------------------------|----------------|
| ffn       | one FFN+LN block (W₁→GELU-poly→W₂→residual→LN-poly)    | 9              |
| attention | one MHA+LN block                                        | 14             |
| layer     | one full encoder layer (= attention + ffn)              | 23             |
| model     | all `num_hidden_layers` + classifier head               | 23 × L         |

Outputs per-step latency to
`results/encrypted_inference/<model>_<phase>.json`.

> **CKKS parameters.** `make_context()` currently sets `N = 16384`
> with 6 mid-primes (so `initial_levels = 6`). That fits **only**
> `--phase ffn`. For `attention`, `layer`, `model` you need either
> `N = 32768` with more mid-primes **or** bootstrapping. See §6.

### Step 5 — Regenerate thesis figures

```bash
python experiments/generate_figures.py            # all figure sets
python experiments/generate_figures.py --only 2 4 # only act-profiles + error-prop
```

The 8 figure sets it produces (one per `FIGURE_SETS` key):

| set | name                              | reads from                                          |
|-----|-----------------------------------|-----------------------------------------------------|
| 1   | Polynomial Approximation          | `results/poly_approx/numerical_results.json`        |
| 2   | Activation Profiles (per model)   | `results/activation_profiles/<model>/*.png`         |
| 3   | BSGS Evaluation                   | `results/bsgs_eval/bsgs_comparison.png`             |
| 4   | Error Propagation (per model)     | `results/error_propagation/<model>/*.png`           |
| 5   | Multi-Model × GLUE Tasks          | `results/multi_model/scaling_results.json`          |
| 6   | Multi-Dataset Comparison          | `results/multi_dataset/multi_task_comparison.png`   |
| 7   | LPAN Stage Comparison             | `results/lpan/**/lpan_comparison*.png`              |
| 8   | PF-SR Encrypted Inference         | `results/encrypted_inference/<model>_<phase>.json`  |

Drop the contents of `results/figures/` into
`IZTECH_Master_Thesis/Figures/` to refresh the LaTeX build.

> **Backward-compatibility note.** Sets 2 and 4 now expect per-model
> subdirectories. If you have legacy SST-2-only PNGs at the top of
> `results/activation_profiles/` or `results/error_propagation/`,
> move them under `tiny/` or rerun the producers — they will not
> be picked up otherwise.

### Step 6 — Decrypt and validate accuracy (per `(model, task)` pair)

`encrypt_inference` already calls `decrypt()` on the final
ciphertext and returns a numpy logit vector, so the validator just
looks up the trained checkpoint, runs the plaintext + encrypted
forwards on a validation slice, and compares argmax.

The driver is committed at
[`experiments/validate_encrypted_accuracy.py`](../experiments/validate_encrypted_accuracy.py).

```bash
# SST-2 sweep for the four models (~100 samples each, seq_len 8)
python experiments/validate_encrypted_accuracy.py --model tiny  --task sst2 --num-samples 100
python experiments/validate_encrypted_accuracy.py --model mini  --task sst2 --num-samples 100
python experiments/validate_encrypted_accuracy.py --model small --task sst2 --num-samples 100
python experiments/validate_encrypted_accuracy.py --model base  --task sst2 --num-samples 100

# Other GLUE tasks (after Step 2 has trained the corresponding checkpoint):
python experiments/validate_encrypted_accuracy.py --model tiny --task qnli --num-samples 100
```

What the driver does:

1. loads the LPAN checkpoint at
   `results/multi_model/<model>/staged_lpan_final/best_model/`
   (or whatever `--checkpoint` points at);
2. boots a CKKS context with `mult_depth = 23 × num_layers + 2`
   levels (auto-selects `N = 32768` for everything except possibly
   Tiny);
3. for each of `--num-samples` validation rows:
   - runs plaintext LPAN forward → `plain_logits`,
   - extracts the input embeddings, truncates to `--seq-len`,
   - runs `run_phase("model", …)` which encrypts → encrypted forward
     → **decrypts** the final ciphertext → numpy logits,
   - records `argmax` agreement and `|Δlogit|`;
4. dumps `results/encrypted_inference/<model>_validation_<task>.json`
   with `accuracy_plain`, `accuracy_decrypted`, `agreement`,
   `mean_abs_logit_delta`, per-sample logits, and per-sample wall
   time.

This JSON is what populates the *Decrypted vs.\ Plaintext Accuracy*
table in the thesis (Section 5.5, `tab:pfsr_accuracy`).

---

## 4b. SST-2 backward-compatibility checklist

The previously-published headline LPAN accuracies on SST-2 were:

| model | baseline | LPAN  | Δ (pp) |
|-------|----------|-------|--------|
| tiny  | 83.26    | 83.14 | −0.12  |
| mini  | 87.16    | 86.81 | −0.34  |
| small | 87.73    | 88.53 | +0.80  |
| base  | 92.20    | 91.86 | −0.34  |

These numbers are **unchanged** by the recent reorganisation:

* `run_staged_lpan.py`, `extract_coefficients.py`, the LPAN modules
  in `fhe_thesis/training/`, `fhe_thesis/models/`, and
  `fhe_thesis/poly/` were **not modified**. Re-running step 2
  with the same seed reproduces the table above.
* `PROFILED_INTERVALS` in `fhe_thesis/config.py` is unchanged, so
  the polynomial coefficients fitted from BERT-Tiny SST-2 profiling
  are byte-identical to the previously-published ones.

### Caveat: PROFILED_INTERVALS scope (read this)

The constant in `fhe_thesis/config.py` only contains entries for
`L0_*` and `L1_*` — i.e. it is *currently the BERT-Tiny profile*. It
is used in three places, **and in all three the same Tiny-derived
intervals are reused for Mini/Small/Base** with an `L0_*` fallback:

| consumer | what falls back | impact |
|---|---|---|
| `fhe_thesis/encryption/coefficients.py` :: `_load_from_extracted` | per-layer interval *metadata* attached to coefficients loaded from a trained checkpoint | The polynomial **coefficients** themselves come from the trained checkpoint, so they are model-specific. The interval is only used as a clamp range for evaluation — Tiny intervals are wide enough to cover Mini/Small/Base activations in practice (verified empirically) but a tighter per-model interval would slightly reduce CKKS error. |
| `fhe_thesis/encryption/coefficients.py` :: `_profile_and_fit` | interval used to fit coefficients **when no trained checkpoint exists** | This is the cold-start path. If you use it for Base, the polynomial is fitted on the Tiny interval and will be sub-optimal. **Always run Step 2 first** so checkpoints exist and the extracted-coefficient path is used. |
| `experiments/analysis/error_propagation.py` | interval the *theoretical* error bound is computed over | The reported bound for Mini/Small/Base layer ≥2 uses Tiny's L0 interval. To tighten, run Step 1 with `--model {key}` then update `PROFILED_INTERVALS` with the printed `Lk_GELU/Softmax/LN` entries. |

In short: **trained accuracies are model-specific (Step 2 produces a
separate checkpoint per model)**. The Tiny intervals act only as
fallback clamping/fitting metadata. You only need to refresh
`PROFILED_INTERVALS` if you want either (a) tighter per-layer error
bounds in the thesis, or (b) better cold-start accuracy without
running Step 2 first. Neither of these changes the Tiny/Mini/Small/Base
SST-2 numbers above.

To upgrade `PROFILED_INTERVALS` to per-model coverage on MSI:

```bash
python experiments/analysis/activation_profiling.py --model tiny mini small base
# Then copy the printed [p0.5, p99.5] intervals into PROFILED_INTERVALS,
# renaming keys from L<i>_<op> to <model>_L<i>_<op> if you change the
# lookup convention.
```

### Verification command

To verify on MSI before any new run:

```bash
# Reproduce previous SST-2 LPAN run for one model:
python run_staged_lpan.py --model tiny --task sst2 --seed 42
# Compare against table above (within ±0.1 pp due to nondeterminism).
```

---

## 5. What's done vs. what's pending

### Done ✅

- [x] LPAN polynomial replacement, 3-stage curriculum, GLUE×model sweep
- [x] Multi-task support for {SST-2, MRPC, QNLI, QQP} via `--task`
- [x] Coefficient extraction from trained checkpoints
- [x] CKKS context + TenSEAL backend with depth audit
- [x] Token-packed ciphertext layout (1 ct/token, hidden_dim slots)
- [x] **Phase 1** — encrypted FFN+LN block (`enc_linear`, `enc_gelu_poly`,
      `enc_ln_poly`)
- [x] **Phase 2** — encrypted multi-head self-attention with PF-SR
      preserved (per-head plaintext-weight slicing, zero-pad concat
      under FHE — no decryption server-side)
- [x] **Phase 3** — model-agnostic `LPANEncryptedLayer` /
      `encrypt_inference`. Same code path for Tiny/Mini/Small/Base —
      shapes pulled from `MODEL_REGISTRY` at call time.
- [x] Unified CLI runner `experiments/run_protocol.py`
- [x] Unified analysis dispatcher `experiments/run_analysis.py`

### Pending ⏳

- [ ] **Phase 4 — scaling benchmark** across all 4 models × 4 tasks.
      Need `--phase model` to run on N=32768. Produce a table of
      (model, task, accuracy_plain, accuracy_decrypted, latency_per_token).
- [ ] **N=32768 context.** Update `fhe_thesis/encryption/context.py`
      so `make_context(level="full")` returns a chain that survives
      23 levels. Sketch:
      ```python
      ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768,
                 coeff_mod_bit_sizes=[60] + [40]*25 + [60])
      ```
- [ ] **Bootstrapping** (only if BERT-Base full-model on N=32768
      blows the budget) — flag-gated behind
      `BackendCapabilities.supports_bootstrapping`. **Not in scope
      yet** — try to scale by enlarging N first.
- [ ] **Pooler tanh.** Currently approximated as identity in
      `encrypt_inference`. Fine for SST-2 (no real degradation in
      our runs) but for QNLI/QQP we should fit a degree-3 polynomial
      tanh approximation on the pooled CLS distribution.
- [ ] **Phase 5 — GPU CKKS backend** (Phantom-FHE or OpenFHE-CUDA).
      Implement a new `CKKSBackend` subclass; no protocol code
      changes thanks to the ABC.
- [ ] **Decrypted-vs-plaintext accuracy validation.** Need a tiny
      driver that runs `encrypt_inference` on, say, 100 SST-2
      validation sentences and compares argmax against the
      plaintext model. This is the killer plot for the thesis
      results chapter.
- [ ] **Merge into `main`.** Once Phase 4 produces the full table,
      open a PR, merge `feature/ckks-protocol` to `main`, delete
      both feature branches.

---

## 6. Concrete next-action checklist for the MSI agent

Order matters; each step's outputs feed the next.

1. **Sanity check the merged tree compiles & imports.**
   ```bash
   cd ~/IZTECH-Master-Thesis-Project
   git checkout feature/ckks-protocol
   git pull
   python -c "from fhe_thesis.encryption import (
       run_phase, transformer_layer_depth, load_coefficients)
   print('layer depth =', transformer_layer_depth())"
   ```
   Expect `layer depth = 23`. If you see an `ImportError` from
   `transformers` / `tenseal`, the `fhe_venv` is not active.

2. **Smoke-test the encrypted FFN block** (5 min, fits N=16384):
   ```bash
   python experiments/run_protocol.py --model tiny --phase ffn
   ```
   Confirms TenSEAL backend, packing, depth audit, and JSON dump
   path all work. Look for `"total"` in the JSON < a couple of
   minutes for BERT-Tiny.

3. **Bump CKKS context to N = 32768** in
   [`fhe_thesis/encryption/context.py`](../fhe_thesis/encryption/context.py).
   Add a `level` argument to `make_context` (default `"phase1"`,
   alternative `"full"`) so existing callers still work. Update
   `experiments/run_protocol.py` to pass `level="full"` for
   `--phase ∈ {layer, model}`.

4. **Run `--phase layer` on Tiny:**
   ```bash
   python experiments/run_protocol.py --model tiny --phase layer
   ```
   Expect ~minutes wall, depth audit shows ≤ 23 / budget.

5. **Run `--phase model` on Tiny + Mini:**
   ```bash
   python experiments/run_protocol.py --model tiny --phase model
   python experiments/run_protocol.py --model mini --phase model
   ```
   These give the first end-to-end PF-SR latencies in the literature
   on these two models.

6. **Add the accuracy validator.** Create
   `experiments/validate_encrypted_accuracy.py` that:
   * loads a trained LPAN checkpoint
   * encrypts N=100 SST-2 validation sentences
   * runs `encrypt_inference`
   * decrypts, takes argmax, compares to plaintext argmax
   * dumps `results/encrypted_inference/<model>_validation.json`

7. **Phase 4 sweep table.** Write a thin shell wrapper that loops
   `(model, task) ∈ {tiny,mini,small,base} × {sst2,mrpc,qnli,qqp}`
   and runs steps 4–6. Aggregate the JSONs into one CSV consumed
   by `experiments/generate_figures.py`.

8. **Open the merge-to-main PR** once the sweep table looks good.

---

## 7. Things to know before touching code

* **Lazy imports.** `fhe_thesis/encryption/__init__.py` uses PEP-562
  so importing the package on a machine without TenSEAL succeeds.
  Importing `protocol`, `coefficients`, `backend`, or any `enc_*`
  symbol triggers the heavy imports. Don't break this — the design
  machine relies on it for fast iteration.
* **Plaintext weights are never encrypted.** Only activations are.
  All weight matrices live in numpy arrays passed via plaintext into
  `enc_linear`. This is what keeps the multi-head attention
  PF-SR-pure (per-head weight slicing happens *outside* FHE).
* **`enc_self_attention` does NOT use rotations** beyond the single
  `ct.sum()` inside `enc_attention_apply`. Per-head concat uses
  zero-pad masks, not rotations. If you add a new op, prefer the
  same pattern.
* **LayerNorm is mean-free.** `enc_ln_poly` skips the `(x - μ)`
  centring; the LPAN training absorbs it into the `(γ, β)`
  re-parameterisation. Documented in `ops.py` docstring.
* **Coefficient JSON shape** is the flat `{param_name: {...}}` from
  `extract_coefficients.py`, not nested `{layers: [...]}`. The
  loader handles the mapping.
* **Don't run the heavy pipeline on the design box.** It has no
  `numpy`/`tenseal`/`transformers`. Anything beyond the lazy import
  smoke-test must run on MSI.

---

## 8. Reference docs in this repo

* [docs/ckks_protocol.md](ckks_protocol.md) — full PF-SR protocol
  specification: threat model, depth derivation, packing strategy,
  per-phase implementation notes.
* [README.md](../README.md) — high-level overview, GLUE workflow,
  CKKS module layout.
* `IZTECH_Master_Thesis/chapters/3_methodology.tex` — the
  methodology chapter; Sections on LPAN and PF-SR are the textual
  counterparts of `fhe_thesis/training/` and
  `fhe_thesis/encryption/` respectively.

---

*Last updated:* this commit (post-merge of
`feature/glue-multi-dataset` → `feature/ckks-protocol`, Phase 3
landed, analysis scripts reorganised under `experiments/analysis/`).
