---
name: vqe-circuits
description: >
  How the eigenstate-preparation pipeline is built from circuits: gate-based
  statevector simulation, the Pauli-rotation primitive, gate-level UCCSD and the
  hardware-efficient ansatz, the prolongation operator, adiabatic resolution
  refinement, and the rodeo algorithm. Use this skill WHENEVER the task involves
  VQE, UCCSD/HEA circuits, prolongation, resolution refinement, the adiabatic
  path, OR the rodeo algorithm (energy scans, eigenstate preparation, the
  cos^2 success probability, fidelity-vs-M, acceptance). Read `references/rodeo.md`
  before writing or modifying any rodeo code so the formulas and conventions
  match the published algorithm.
---

# VQE circuits, resolution refinement, and the rodeo algorithm

The four pipeline stages, all implemented in `pairinglib` and reused (never
re-implemented) by the notebook and analysis scripts.

## 1. Coarse-state preparation (gate-level UCCSD-VQE)
- Trotterised product ansatz `prod_mu exp(theta_mu (tau_mu - tau_mu^dag)) |HF>`.
- Each excitation compiles to commuting Pauli rotations via Jordan-Wigner: a
  single -> 2 Pauli strings, a double -> 8. Use `gate_uccsd_setup/_vqe/_state`.
- Optimise with the EXACT analytic product gradient. The naive two-term pi/4
  parameter shift is WRONG beyond the first factor (later factors gain
  frequency-1 cross terms; the spectrum is {1,2}). This is a known trap — always
  use the analytic gradient already implemented in the library.
- Singles vanish for the pairing force (seniority): expect a doubles theory.

## 2. Prolongation
- `prolong_matrix(k_low, k_high, N)` embeds every low-res basis state into the
  same occupation in the larger space, leaving new levels empty: gate-free for
  basis refinement, `O(N_sites)` two-qubit gates per dimension for lattices.
- The prolonged state satisfies `<Phi0|H_high|Phi0> = E_low` exactly.

## 3. Resolution refinement (adiabatic phase)
- Path `H(s) = cos^2(pi s/2) P (H_low - mu) P^dag + sin^2(pi s/2)(H_high - mu)`,
  `s = t/T`, with shift `mu = E_low + 0.6` to keep tracked states below the large
  null space of `P^dag`. Use `refine_state` / `refine_NK`.
- The path stays gapped, so `T ~ 1/Delta E` (not `1/Delta E_min^2`). Overlap
  rises toward 1; energy falls to FCI from above.

## 4. Rodeo algorithm (final filter) — see references/rodeo.md
- One cycle: ancilla Hadamard, controlled `e^{-iHt_k}`, phase `P(E t_k)`,
  Hadamard, measure; post-select all-zero. The post-selected map is
  `(I + e^{iEt} e^{-iHt})/2` (`rodeo_post0`), equal to the explicit ancilla
  circuit (`rodeo_cycle_ancilla`).
- Feed the rodeo the RESOLUTION-REFINED state (high overlap p): that is the
  whole point — cost ~ 1/p and a few cycles when p ~ 1.

## Validation anchors (assert these when you touch the code)
- `rodeo_cycle_ancilla` == `rodeo_post0` to ~1e-16.
- Eigenstate input reproduces `prod cos^2(t_k (E_obj-E)/2)` (Eq. 1); Gaussian
  average reproduces `[(1+e^{-(E_obj-E)^2 sigma^2/2})/2]^M` (Eq. 2).
- Refined input (N=4, k=4, p~0.998) -> fidelity ~ 0.999995 within ~2 cycles,
  acceptance -> p; off-target background ~ `2^-M`.
