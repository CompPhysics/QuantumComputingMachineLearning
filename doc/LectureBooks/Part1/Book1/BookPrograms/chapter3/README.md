# Programs for chapter 3

Every number, table and figure of chapter 3 is produced by the programs in this folder together with the shared modules in `../common/` that this chapter introduces (`fockspace.py`, `models.py`, `matrixfunctions.py`, `optimize.py`). Keep this list and the `Programs for this chapter` section of `chapter3.tex` in sync. Figures are written to `../../BookFigures/chapter3/`.

| Program | What it computes | Runtime |
|---|---|---|
| `second_quantization.py` | Bit-string determinants and signs, anticommutators as matrices, Wick's theorem against brute force, Slater–Condon rules, the normal-ordered Hamiltonian (reference energy, Fock matrix, Brillouin elements), a two-mode Hamiltonian | instant |
| `fci.py` | Exact diagonalisation of the Lipkin, pairing, Hubbard and Heisenberg models (Tables 3.lipkin–3.wall), truncated CI (CIS/CID/CISD), the size-consistency failure of CID, Fig. 3.models | ~10 s |
| `hartreefock.py` | `HartreeFock` class: SCF with density matrices, Brillouin check, Thouless states, stability matrix; Lipkin HF and its instability at chi = 1, pairing (HF does nothing), Hubbard UHF; Fig. 3.lipkin_hf | seconds |
| `trotter.py` | BCH series order by order, Lie–Trotter and Strang splitting for X + Z and for the Heisenberg ring (even/odd bonds); Fig. 3.trotter | instant |
| `coupledcluster.py` | `CCD` class (amplitude equations, MP2 start) and `UnitaryCC` (exact and Trotterised exponentials, BFGS optimisation) for the pairing model: Tables 3.ccd, 3.uccd, 3.ucctrotter; Fig. 3.coupled_cluster | ~10 s |

All programs run on `numpy` and `matplotlib` only; each runs as a script with `assert`-based checks and reproduces the numbers printed in the chapter to every digit shown.
