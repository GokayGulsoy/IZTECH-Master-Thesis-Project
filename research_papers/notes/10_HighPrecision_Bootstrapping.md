# High-Precision Bootstrapping of RNS-CKKS via Inverse-Sine + Minimax

**Citation**: Lee, Lee, Lee, Kim, No. *EUROCRYPT 2021* (extended in IEEE TIT 2021). Seoul National University.

**One-line summary**: Improves bootstrapping precision in RNS-CKKS from ~20 bits to **32.6–40.5 bits** by using minimax-optimal polynomial approximation of mod-reduction + inverse-sine composite function.

## Core contribution

- **Improved multi-interval Remez algorithm** — derives the optimal minimax polynomial of any continuous function over any union of intervals.
- **Composite inverse-sine method** — narrows the gap between bootstrapping scaling factor and default scaling factor by composing the standard mod-reduction with `arcsin` polynomial.
- Reduces bootstrapping approximation error by **42× to 1176×** (i.e., 5.4–10.2 bits of precision improvement per parameter set).
- Final achievable precision: **32.6–40.5 bits** with composite method (vs. 27.2–30.3 without).

## Relevance to HyPER-LPAN

- Bootstrapping precision is the **floor** of our end-to-end accuracy budget. With ~40-bit bootstrap precision, we have headroom for layer-by-layer accumulated noise across our 25-depth circuit and 12 transformer layers.
- Justifies our depth-25 choice with bootstrap budget [3, 3]: fresh ciphertext gives L − K levels, and high-precision bootstrap means each refresh barely degrades accuracy.
- The minimax / Remez framework is the same one we cite in §3.3 of the thesis for our LPAN polynomial design.

## Direct citation use

- "We rely on the high-precision bootstrapping of Lee et al. [EUROCRYPT'21], which provides ~40-bit precision per refresh — sufficient for the cumulative noise budget of our 12-layer composition with depth 25 between bootstraps."
