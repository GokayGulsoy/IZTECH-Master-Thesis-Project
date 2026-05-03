# Universal Transformers

**Citation**: Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser. *ICLR 2019*. University of Amsterdam + DeepMind + Google Brain.

**One-line summary**: Combines transformer parallelism with RNN-style **recurrence in depth** (parameters tied across layers) plus a **dynamic per-position halting mechanism** (ACT). Turing-complete under certain assumptions.

## Core contribution

- **Parameter-tied** self-attention + transition function applied recurrently in time.
- **Adaptive Computation Time (ACT)** per position: each token decides when to halt.
- Outperforms vanilla Transformer on algorithmic tasks (string copying, logical inference) and language modeling (LAMBADA SOTA at the time).

## Key numbers

- WMT14 En-De: **+0.9 BLEU** over Transformer.
- LAMBADA: new SOTA.
- Generalizes to longer-than-training sequences on copy / logic tasks.

## Relevance to HyPER-LPAN

- **Adaptive computation per position** is the spiritual ancestor of PoWER-BERT and Length-Adaptive Transformer.
- **Parameter tying across layers** is the same idea ALBERT later popularized — and it's the simplest way to train a Universal Transformer for FHE: train once, evaluate any number of recurrent steps.
- The **per-position halting** is dynamic and **not FHE-compatible** (early-exit reveals computation depth ↔ data leakage).
- We could borrow the **parameter-tying** idea to dramatically reduce the model size we need to encrypt.

## Direct citation use

- "Adaptive computation in transformers dates back at least to Universal Transformers [Dehghani et al., ICLR'19], which introduced per-position halting via ACT. Such dynamic mechanisms cannot be evaluated under FHE without leaking computation paths; HyPER-LPAN therefore commits to **input-independent** layer configurations chosen offline by the composition selector."

## Future-work hook

- Universal-Transformer-style parameter tying could halve the encrypted weight storage. Worth exploring in §7.
