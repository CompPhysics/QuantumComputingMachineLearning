# Programs for chapter 2

Every number, table and figure of chapter 2 is produced by the programs in this folder together with the shared modules in `../common/` that this chapter introduces (see `../../CLAUDE.md` and the `qc-book-code` skill). Keep this list and the `Programs for this chapter` section of `chapter2.tex` in sync. Figures are written to `../../BookFigures/chapter2/` as PDF (for LaTeX) and PNG (for the notebook).

| Program | What it computes | Runtime |
|---|---|---|
| `dirac_algebra.py` | Worked examples of Sec. 2.2: inner/outer products, projectors, Pauli algebra, tensor products, spectral decomposition | instant |
| `bloch_sphere.py` | Bloch angles and vectors, Table 2.1, Fig. 2.1 (Pauli eigenstates, Larmor precession, rotations) | seconds |
| `measurement.py` | Born rule, post-measurement states, X-basis measurement, marginals of a 3-qubit register, Fig. 2.2 (shot noise) | seconds |
| `decoherence.py` | Dephasing and amplitude-damping channels (Kraus form), Fig. 2.3 | instant |
| `entanglement.py` | Bell states, measurement correlations, partial traces, Schmidt decomposition via SVD, Table 2.2, Fig. 2.4 | instant |
| `protocols.py` | Superdense coding (all four messages) and teleportation (200 random states) on the simulator | instant |
| `two_level_system.py` | Two-level Hamiltonian in matrix and Pauli form, Table 2.3, grid search over the R_y R_x ansatz, Fig. 2.8 | seconds |

Shared modules introduced by this chapter (in `../common/`): `gates.py`, `statevector.py`, `circuit.py`, `density_matrix.py`, `bookplot.py`. All programs run on `numpy` and `matplotlib` only; each runs as a script with `assert`-based checks.
