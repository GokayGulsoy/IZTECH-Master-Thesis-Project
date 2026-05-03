# Length-Adaptive Transformer — Train Once with LengthDrop, Use Anytime with Search

**Citation**: Kim, Cho. *ACL 2021*. NAVER Clova AI + NYU. Code: github.com/clovaai/length-adaptive-transformer.

**One-line summary**: Extends PoWER-BERT with **LengthDrop** (random per-layer length per SGD step) and **multi-objective evolutionary search** to give a **single model** usable at any inference-time length budget — plus **Drop-and-Restore** to support token-level tasks.

## Core contribution

1. **LengthDrop**: at each SGD step, sample a length configuration `(l1, …, lL)` where `l(i+1) ~ U((1-p)·li, li)`. Like nested dropout for sequence length. Single trained model contains many sub-models.
2. **Evolutionary search** over length configurations: produces full **Pareto frontier** of accuracy-vs-efficiency without retraining.
3. **Drop-and-Restore**: vectors are *set aside* (not deleted) in intermediate layers and restored at the final layer. Enables span-based QA (SQuAD) — fixes PoWER-BERT's sequence-level-only limitation.
4. Uses **LayerDrop** (Fan et al.) jointly to make word-vectors agnostic to skipping layers.

## Why it matters

PoWER-BERT requires one model per budget; LAT trains once, selects at inference time. Critical for deployment where different users have different latency budgets.

## Relevance to HyPER-LPAN

- The **multi-objective evolutionary Pareto search** is conceptually parallel to our composition selector — both are searching over per-layer configurations to optimize an accuracy/cost trade-off.
- LAT searches **lengths**; our MCKP-DP searches **primitives × lengths**. We could combine: extend the MCKP axis to include "keep ratio" giving a 4-dimensional search (LM/Q/L × keep-ratio).
- **Drop-and-Restore** is interesting if we ever target token-level tasks under FHE (NER, QA).
- **Not directly FHE-compatible** (dynamic shapes). Same limitation as PoWER-BERT.

## Direct citation use

- "Length-Adaptive Transformer [Kim & Cho, ACL'21] extends PoWER-BERT to train a single network usable across length budgets via LengthDrop and evolutionary search. Our composition selector takes a related multi-objective view (MCKP-DP over primitive choice) but is solved exactly because the per-layer cost units are integer-valued."

## Comparison points

| | PoWER-BERT | LAT | HyPER-LPAN MCKP |
|---|---|---|---|
| Search axis | length per layer | length per layer | primitive per layer |
| Training | per-budget retrain | single model + search | single trained model + DP |
| Search method | gradient relaxation | NSGA-II evolutionary | exact dynamic programming |
| Pareto frontier | one point per train | full curve | full curve (sweep budget) |
| FHE-compatible | no (dynamic shape) | no (dynamic shape) | **yes** (static config) |
