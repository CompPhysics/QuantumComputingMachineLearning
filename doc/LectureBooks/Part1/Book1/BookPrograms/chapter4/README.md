# Programs for chapter 4

Every number, table and figure of chapter 4 is produced by the programs in this folder together with the shared modules `../common/pauli.py` and `../common/jordanwigner.py` that this chapter introduces. Keep this list and the `Programs for this chapter` section of `chapter4.tex` in sync. Figures are written to `../../BookFigures/chapter4/`.

| Program | What it computes | Runtime |
|---|---|---|
| `jw_basics.py` | Failure of the naive map, the Jordan–Wigner identities (number, hopping, pair, eight-string two-body term), Table 4.paulicount (6, 19, 55, 136, 292 strings), the two-qubit Hamiltonian, Fig. 4.pauliweights | ~1 s |
| `mapped_models.py` | Lipkin model in four encodings (Table 4.lipkinencodings), pairing model per orbital and per level (Eqs. 4.jwpairingfull, 4.jwpairingtwolevels, Table 4.paulicounts), Hubbard ring in two orderings, Heisenberg ring and its fermionic image; every spectrum checked against chapter 3; Fig. 4.encodings | ~1 s |
| `pauli_circuits.py` | Circuits for exp(-i theta P), Trotter steps as gate sequences with counts and errors, energy estimation from shots with qubit-wise commuting groups, the pair-excitation gate, a four-parameter UCC circuit reproducing the UCCD energies of chapter 3; Fig. 4.circuits | ~10 s |

All programs run on `numpy` and `matplotlib` only; each runs as a script with `assert`-based checks.
