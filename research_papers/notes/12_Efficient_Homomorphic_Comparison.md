# Efficient Homomorphic Comparison Methods with Optimal Complexity

**Citation**: Cheon, Kim, Kim. *ASIACRYPT 2020*. Seoul National University.

**One-line summary**: Composite-polynomial approximation of the sign function with **optimal `Θ(log(1/ε)) + Θ(log α)`** complexity for ε-precision α-bit comparison.

## Core contribution

- Identifies **core properties** a constant-degree polynomial `f` must satisfy so that `f ∘ f ∘ … ∘ f` converges to the sign function.
- Acceleration: **mixed composition** `f ∘ … ∘ f ∘ g ∘ … ∘ g` with two polynomials of complementary properties.
- 20-bit integer comparison at α=20 in **1.43 ms amortized** — **30× faster** than prior work.
- Improves on Cheon et al. (ASIACRYPT 2019) `comp(a,b) = lim ak/(ak + bk)` from quasi-optimal to optimal asymptotic.

## Why it matters

Comparison and sign functions underpin:
- Argmax (NEXUS uses `O(log m)` sign operations).
- ReLU.
- Sorting.
- Decision-tree evaluation.

Without an efficient comparison primitive, classification heads under FHE are intractable.

## Relevance to HyPER-LPAN

- Used inside our final classification head (per NEXUS recipe).
- Provides the precision/depth budget we should plan around for any sign-based primitive (e.g., if we add a hard-attention variant in future).
- Cite in §3 (FHE primitives) alongside CKKS, bootstrapping, and minimax.

## Direct citation use

- "Sign-based primitives such as Argmax are evaluated using the composite-polynomial comparison method of Cheon et al. [ASIACRYPT'20], which achieves optimal `Θ(log(1/ε)) + Θ(log α)` complexity."
