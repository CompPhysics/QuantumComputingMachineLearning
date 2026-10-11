# common

Helpers imported by programs of several chapters. Every `BookPrograms/*` folder is put on `sys.path`, so `from statevector import StateVector` works from any chapter.

| Module | Contents | Introduced in |
|---|---|---|
| `gates.py` | Gate matrices (`I X Y Z H S T`, `Rx Ry Rz Rn`, `phase`, `U3`, `CNOT CZ SWAP TOFFOLI`, `controlled(U)`), `commutator`, `kron`, unitarity checks | chapter 2 |
| `statevector.py` | `StateVector`: 2^n amplitudes, `apply(U, targets)` by tensor contraction, gate shorthands, `marginal`, `measure`, `sample`, `expectation`, `density_matrix` | chapter 2 |
| `circuit.py` | `Gate` and `Circuit`: gate lists with controls, `run`, `unitary`, `depth`, `inverse`, ASCII `draw`, `bell_circuit` | chapter 2 |
| `density_matrix.py` | `DensityMatrix`: ensembles, Bloch vector, purity, `measure`, `evolve`, `partial_trace`, `von_neumann_entropy`; `shannon_entropy` | chapter 2 |
| `bookplot.py` | Figure directory `BookFigures/chapterN/`, matplotlib defaults, `savefig(fig, chapter, name)` writing PDF + PNG | chapter 2 |
| `fockspace.py` | `FockSpace(n)`: 2^n-dimensional Fock space with `create/annihilate/number` matrices (Z-string form), `one_body(h)`, `two_body(v)`, `hamiltonian`, `sector(N)`; `SlaterBasis(n, N)`: determinants as bit strings, `apply(ops, state)` with signs, `hamiltonian_matrix(h, v)` | chapter 3 |
| `models.py` | `LipkinModel`, `PairingModel`, `HubbardModel`, `HeisenbergModel`: `(h, v)` in a spin-orbital basis plus the symmetry-adapted blocks (`quasispin_matrix`, `seniority_zero_matrix`, `sector_matrix`); `add_term` for antisymmetrised tensors | chapter 3 |
| `matrixfunctions.py` | `expm_hermitian(H, scale)`, `expm_antihermitian(A)`, general `expm` by scaling and squaring | chapter 3 |
| `optimize.py` | `minimize(f, x0)`: BFGS with Armijo backtracking and numerical gradients (no scipy) | chapter 3 |
| `pauli.py` | `PauliSum`: dictionary `{string: coefficient}` with products, commutators, `from_matrix` (trace decomposition), `matrix`, `weights`, `commuting_groups`, `expectation` | chapter 4 |
| `jordanwigner.py` | `JordanWigner(n)`: `create/annihilate/number` as `PauliSum`s with the parity string, `map_hamiltonian(h, v)`; `ParityMapping` for comparison | chapter 4 |

Qubit 0 is the leftmost label / most significant bit of the basis-state index throughout (so `|01>` has index 1 and `np.kron(a, b)` matches `|a>|b>`). Mode 0 of `FockSpace` is the same leftmost factor, `|1>` means occupied, and `a_p = Z ⊗ ... ⊗ Z ⊗ σ⁻ ⊗ I ⊗ ... ⊗ I` — the Fock-space matrices of chapter 3 *are* the Jordan–Wigner images of chapter 4. Sign conventions used everywhere: `σ⁺ = (X − iY)/2 = |1><0|`, `n_p = (I − Z_p)/2`; the Lipkin lower level and the empty pairing level are `|0>`.
