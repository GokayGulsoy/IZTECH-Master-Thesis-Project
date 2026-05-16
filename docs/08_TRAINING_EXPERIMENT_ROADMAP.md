# 08 — Training And Experiment Roadmap

Purpose: lock the pre-pod execution order so H100 time is spent on training and measurement, not on deciding what to try next.

## Current decision summary

- **Keep Synthesizer-LPAN as the mainline.**
- **Do not pivot** to Linformer / Performer / Nyströmformer / AFT before the core task sweep.
- **First follow-up improvement** after the base task sweep: compress the exported Synthesizer pattern $A_h$ rather than replacing the entire architecture.
- **First future architecture branch** after the mainline is complete: FNet on tiny/mini only.

## Supported task surface

The current code supports the following GLUE tasks in [fhe_thesis/tasks.py](../fhe_thesis/tasks.py):

- `sst2`
- `mrpc`
- `rte`
- `qnli`
- `qqp` (supported, but notably more expensive)

Recommended order:

1. `sst2` — easiest full-pipeline lock and best place to validate real Stage-4 export + FHE bench.
2. `mrpc` — small sentence-pair task; checks F1-sensitive behavior and pair-tokenization path.
3. `rte` — hardest low-data task; most sensitive architecture stress test.
4. `qnli` — larger sentence-pair task; stabilizes generalization claims.
5. `qqp` — optional only if budget remains.

## Pod gating rules

Only promote a task to expensive H100 benchmarking if it passes the stage gates below.

### Baseline gate

- Baseline fine-tune must land within about **0.5--0.8 points** of the stored expected plaintext baseline for that task.
- If baseline misses badly, stop and fix the data/training surface before Stage 1.

### Stage-3 gate

- `sst2`, `qnli`: Stage 3 should stay within about **2.0 points** of baseline on the primary metric.
- `mrpc`, `rte`: Stage 3 should stay within about **2.5 points** of baseline on the primary metric.
- If Stage 3 fails, do not run Stage 4 yet; repair the polynomial-training chain first.

### Stage-4 gate

- `sst2`, `qnli`: Stage 4 should lose no more than about **1.0 extra point** versus Stage 3.
- `mrpc`, `rte`: Stage 4 should lose no more than about **1.5--2.0 extra points** versus Stage 3.
- If Stage 4 fails, run Stage-4 ablations before moving to the next task.

### FHE benchmark gate

Only export + benchmark tasks whose Stage-4 checkpoints pass the Stage-4 gate.

## Exact per-task run order

The command mechanics are already documented in [05_REPRODUCING_RESULTS.md](05_REPRODUCING_RESULTS.md). For each task, the operational order is:

1. `python -m fhe_thesis.training.run_staged_lpan --model base --task <task> --stage all`
2. `python -m fhe_thesis.training.run_synth_lpan --model base --task <task> --stage 4`
3. Evaluate the Stage-4 summary and compare against the task gate.
4. If passed: `python -m fhe_thesis.training.export_synth_lpan --checkpoint-dir results/synthesizer_lpan/<task>/base/best_model`
5. Run the HE benchmark on the exported real checkpoint.

## Tiered H100 priority

### Tier 1 — mandatory before thesis writing stabilizes

1. `sst2` base, full Stage 1→4 chain.
2. `sst2` export + real-checkpoint HE benchmark.
3. `mrpc` base, full Stage 1→4 chain.
4. `mrpc` export + real-checkpoint HE benchmark if Stage 4 is healthy.

### Tier 2 — strong thesis coverage

1. `rte` base, full Stage 1→4 chain.
2. `qnli` base, full Stage 1→4 chain.
3. Export + HE benchmark for whichever of `rte` / `qnli` passes most cleanly.

### Tier 3 — only if budget remains

1. `qqp`.
2. Pattern-compression recovery fine-tunes.
3. Tiny/mini future-architecture pilots such as FNet.

## Must-run experiments after training

### A. Core stage-by-stage accuracy table

For each trained task, report:

- baseline
- Stage 1
- Stage 2
- Stage 3
- Stage 4

This is mandatory because it localizes where accuracy is lost.

### B. Stage-4 ablations

Run these on `sst2` first, then only promote the winner to other tasks:

1. Teacher-average initialization of $A_h$ vs random initialization.
2. Annealed Stage-4 KD weight vs fixed KD weight.
3. Optional freeze policy ablation if needed: train only Synthesizer parameters vs train Synthesizer + nearby linear layers.

### C. Attention-pattern diagnostics

For the trained Stage-4 model, collect per-layer/per-head statistics:

- entropy
- diagonal mass concentration
- effective rank / singular-value decay
- head-to-head similarity

These diagnostics decide whether pattern compression is likely to work.

### D. Pattern-compression ablations

Run on the best `sst2` Stage-4 checkpoint first:

1. top-k diagonal pruning by absolute mass
2. fixed band-limited masks
3. low-rank SVD approximation of $A_h$

Recommended order:

- First do **zero-retrain** compression to estimate raw latency/accuracy sensitivity.
- Then take the best candidate and run a **short recovery fine-tune**.

### E. Real-checkpoint FHE benchmark matrix

For every task that passes the gate, bench the exported real checkpoint with:

- naive Synthesizer vs BSGS
- `BATCH=1,4,8,16` if memory permits
- `chain=22` as default, `chain=24` only when numerically necessary

Use the same exported checkpoint path so the measurement remains directly tied to the trained model.

## Pod time allocation

Recommended order for the first serious H100 session:

1. Environment bring-up and HEonGPU/bindings build.
2. `sst2` base full chain.
3. `sst2` export + benchmark.
4. `mrpc` base full chain.
5. Stop and review before launching `rte` / `qnli`.

Reason: `sst2` validates the whole stack, while `mrpc` is the cheapest cross-task generalization check.

## Deferred until after the core sweep

- FNet architecture branch.
- Factorized/low-rank Synthesizer redesign.
- Dynamic token pruning variants inspired by PoWER-BERT / Length-Adaptive Transformer.
- PEGASUS-style mixed-crypto non-linear handling.

These are future branches, not blockers for the current thesis-critical training matrix.