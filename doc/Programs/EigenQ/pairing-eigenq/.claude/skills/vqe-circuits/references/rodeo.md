# Rodeo algorithm — formulas and conventions

Reference: K. Choi, D. Lee, J. Bonitati, Z. Qian, J. Watkins, PRL 127, 040505
(2021); Z. Qian et al., Eur. Phys. J. A 60, 151 (2024) (+ supplemental).

## One cycle (object register + 1 reused ancilla)
1. Hadamard on ancilla.
2. controlled-`e^{-iH t_k}` on the object (control = ancilla).
3. phase `P(E t_k)` on the ancilla (this shifts H -> H - E).
4. Hadamard on ancilla; measure; reuse via mid-circuit measurement.

Post-selecting ancilla |0> applies `(I + e^{iE t_k} e^{-iH t_k})/2` to the
object; an eigenstate |E_j> picks up `e^{i(E-E_j)t_k/2} cos[(E_j-E)t_k/2]`.

## Success probability
- Eigenstate input, given times:  `P = prod_k cos^2[ t_k (E_obj - E)/2 ]`   (Eq. 1)
- Gaussian times, RMS sigma:      `P = [ (1 + e^{-(E_obj-E)^2 sigma^2/2}) / 2 ]^M`  (Eq. 2)
- On resonance (E = E_obj) every factor is 1; far off resonance the per-cycle
  factor -> 1/2, so the background is `2^-M`. Energy resolution ~ 1/sigma.

## Convergence criterion (state preparation, E tuned to E_0)
- `P_M = p + sum_{k!=0} |c_k|^2 [ (1+e^{-(E_k-E_0)^2 sigma^2/2})/2 ]^M`, `p=|c_0|^2`.
- Fidelity `F_M = p / P_M ~ [1 + (1-p)/p 2^-M]^-1`; infidelity `1-F_M <~ (1-p)/p 2^-M`.
- Cycles for `1-delta`:  `M >~ log2[(1-p)/(p delta)]`.
- Acceptance `P_M -> p`, so repetitions ~ 1/p; total effort `O(|log delta|/(p eps))`.

## On hardware
`e^{-iHt}` is realised by Suzuki-Trotter over the 2k qubits (the rodeo then
filters eigenstates of the effective Trotterised Hamiltonian). The single-qubit
benchmark uses a Z-Y-Z Euler rotation `R_nhat(theta)=e^{-i theta nhat.sigma/2}`.
In simulation we apply the controlled evolution exactly inside the conserved
N-particle sector (see rodeo.py), isolating algorithmic behaviour from Trotter error.
