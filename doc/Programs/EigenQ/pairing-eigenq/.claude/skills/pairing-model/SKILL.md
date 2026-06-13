---
name: pairing-model
description: >
  Physics conventions and the validated Python API for the constant-pairing
  Hamiltonian (nuclear/condensed-matter pairing model). Use this skill WHENEVER
  the task touches the pairing model, seniority-zero spectra, the
  delta/g Hamiltonian, Jordan-Wigner encoding of pairs, Hartree-Fock pairing
  energies, FCI/CCD/UCCSD benchmarks for pairing, or the `pairinglib` package.
  Consult it before writing any pairing physics so the conventions (level
  index, qubit ordering, sign of g, benchmark numbers) stay consistent across
  the notebook, the library, and the paper.
---

# Pairing model: conventions and API

This project studies the constant-pairing Hamiltonian as a controlled testbed
for the resolution-refinement + rodeo eigenstate-preparation pipeline. Always
use the conventions below; they are baked into `src/pairinglib` and the paper.

## Hamiltonian and conventions (do not deviate)
- `H = delta * sum_p p * sum_sigma n_{p,sigma} - (g/2) * sum_{p,q} a+_{p up} a+_{p dn} a_{q dn} a_{q up}`
- `delta = 1` throughout. Level `p` (0-indexed) has single-particle energy `delta*p`.
- Qubit index `= 2*p + sigma`, with `sigma = 0` (up), `1` (down); a `k`-level
  problem uses `2k` qubits. Big-endian: qubit `j` carries bit `(n-1-j)`.
- The force conserves seniority, so the `N`-paired ground state lies entirely in
  the seniority-zero subspace; the full `N`-sector FCI equals the seniority-0 result.
- Hartree-Fock fills the lowest `N/2` levels; `E_HF(N=4, g) = 2 - g`.
- "Basis refinement" means increasing `k` at FIXED `N` (the basis/qubit count
  grows; the particle number does not).

## Validated benchmark numbers (N=4, g=1) — use as regression anchors
| k | qubits | FCI       | CCD       | UCCSD     |
|---|--------|-----------|-----------|-----------|
| 2 | 4      | 1.000000  | 1.000000  | 1.000000  |
| 3 | 6      | 0.794697  | 0.794697  | 0.794697  |
| 4 | 8      | 0.635548  | 0.630443  | 0.636987  |
| 8 | 16     | 0.145559  | 0.097263  | 0.157856  |
UCCSD is variational (>= FCI); CCD over-binds (< FCI). Singles vanish (seniority).

## Library API (import from `pairinglib`)
See `references/library-api.md` for full signatures. Do NOT redefine these in
notebooks or scripts — import them, so there is a single tested source of truth.

## Common pitfalls
- `np.eye(2, dtype=complex)` — the 2nd positional arg of `np.eye` is the column
  count, not the dtype.
- Never shadow stdlib modules with file names (e.g. a file called `re.py`).
- The dense JW pair-interaction operator `W = sum_{p,q} A+_p A_q` is already
  Hermitian; use `H = H_kin - (g/2) W` (do not add `W + W.dag`, that double counts).
