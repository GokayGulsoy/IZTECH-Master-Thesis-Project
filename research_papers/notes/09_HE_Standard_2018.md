# Homomorphic Encryption Standard (2018)

**Citation**: Albrecht, Chase, Chen, Ding, Goldwasser, Gorbunov, Hoffstein, Lauter, Lokam, Micciancio, Moody, Morrison, Sahai, Vaikuntanathan. *Homomorphic Encryption Standardization Workshop, March 2018*.

**One-line summary**: Community-consensus document specifying RLWE-based HE schemes (BGV, B/FV) and recommending concrete parameters for 128 / 192 / 256-bit security.

## Core contribution

- Standardizes **encryption schemes** (BGV and B/FV; also describes YASHE, NTRU/LTV, GSW).
- Specifies **security definitions** for HE.
- Documents **known attacks** on (R)LWE and their estimated runtimes.
- Recommends **concrete parameters** (ring dimension N, ciphertext modulus q, error distribution) for various security levels.
- Serves as a basis for downstream library choices (SEAL, HElib, Palisade, HEAAN, OpenFHE).

## Relevance to HyPER-LPAN

- Justifies our parameter choice: ring dimension `N = 65536`, depth 25, 128-bit security — directly traceable to this standard.
- Cite as the source of our security parameter selection.

## Direct citation use

- "We instantiate RNS-CKKS in OpenFHE 1.2.3 with ring dimension N = 65 536 and a ciphertext modulus chain providing 128-bit security per the Homomorphic Encryption Standard [Albrecht et al., 2018]."

## Note

Some recommendations have been refined by subsequent work (e.g., updated lattice attack estimates, new schemes like CKKS variants). Cite as the foundational standard but be aware that 2018 estimates predate some attacks (lattice estimator updates 2020-2024).
