# pairinglib API reference

## hamiltonian.py
- `build_sector(k, N) -> (nq, states, index)` — fixed-N Fock sector (bitstring list + lookup).
- `H_pairing_sparse(k, g, N, states, index, delta=1.0) -> csr_matrix` — sparse N-sector H.
- `fci_ground(H) -> float` — lowest eigenvalue (exact).
- `E_HF(N, g, delta=1.0) -> float` — Hartree-Fock energy.
- `build_H_full(k, g, delta=1.0) -> ndarray` — dense H over the full 2^(2k) space (HEA/gate-UCCSD).

## ccd.py
- `run_ccd(k, g, N, delta=1.0, ...) -> (E, iters)` — coupled-cluster doubles.

## uccsd.py  (operator-level / statevector)
- `uccsd_pool(k, N) -> (singles, doubles)`
- `setup_uccsd(k, N, delta=1.0) -> dict(nq, states, index, ...)`
- `uccsd_vqe(setup, H, n_trotter=1, ...) -> (E, x, energy_fn, grad_fn)`

## gates.py
- `apply_1q(psi, U, q, n)`, `apply_cnot(psi, c, t, n)`, `Ry(th)`, `Rz(th)`, `Rx(th)`
- `pauli_exp(psi, phi, pauli, n)` — exp(-i phi/2 P) via basis change + CNOT ladder + Rz.

## gate_uccsd.py  (gate-level circuits)
- `make_single(i, a, n)`, `make_double(i, j, a, b, n)` — Pauli terms of an excitation.
- `apply_exc_circuit(psi, theta, terms, n)`
- `gate_uccsd_setup(k, N) -> dict(nq, terms, gens, hf, P, ncnot, nrot)`
- `gate_uccsd_vqe(k, N, g, setup) -> (E, nit)`
- `gate_uccsd_state(k, N, g) -> (E, statevector)`

## refine.py
- `prolong_matrix(k_low, k_high, N) -> ndarray` — gate-free basis embedding.
- `refine_state(N, k_high, T, k_low=2, M=160, mu_buf=0.6, g=1.0) -> (psi, H_high)`
- `refine_NK(N, k_high, Ts, ...) -> dict(ov, en, Eh, El, gap, ov0, E0)`

## rodeo.py
- `rodeo_post0(v, U, t, E) -> (state, prob)` — post-selected one-cycle filter.
- `rodeo_cycle_ancilla(v, U, t, E)` — explicit ancilla circuit (same map).
- `rodeo_sweep(v0, H, ts, E) -> (state, cumulative_acceptance)`
- `rodeo_allzero_prob(v0, H, ts, E) -> float` — for energy scans.
- `rodeo_track(v0, H, ts, E, target) -> (fidelity_array, acceptance_array)`
