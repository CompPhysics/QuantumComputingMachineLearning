# Programs for chapter 5, Quantum Boltzmann Machines

All programs are NumPy + matplotlib only and import the shared modules of
`../common/` (`optimize.py` with `Adam` and `numerical_gradient`,
`bookplot.py`) and the `QNN` class of `../chapter4/qnn.py` (for the
variational thermalizer).  Everything is exact: the models have at most
seven qubits (dense matrices of dimension 128), so every approximation
discussed in the text (contrastive divergence, the clamped-state bound,
the variational Gibbs state) is measured against the truth.  Figures go to
`../../BookFigures/chapter5/` (PDF + PNG).  Every number quoted in
`chapter5.tex` is printed by one of them.

| Program | What it computes | Output | Runtime |
|---|---|---|---|
| `gibbs.py` | Pauli strings, Gibbs states, partial traces, entropies, free energy, relative entropy, fidelity, the Duhamel derivative of `exp(-beta H)` by divided differences in the eigenbasis, the thermofield double | checks of Klein's inequality, Duhamel vs finite differences, TFD; Fig. 5-freeenergy | 2 s |
| `rbm.py` | The classical `RBM` with binary units: exact marginal and gradient, block Gibbs sampling, CD-k, persistent CD; the XOR and bars-and-stripes targets | Table 5-cd, Fig. 5-rbm | 10 s |
| `qbm.py` | The `QBM` class: Hamiltonian from Pauli-string terms, visible marginal, NLL, exact (Duhamel) gradient, clamped-state bound gradient, quantum relative entropy and its gradient, Adam training | self-test, Table 5-gradients | 1 s |
| `classical_data.py` | XOR and bars and stripes learned by the RBM, the QBM without transverse fields, the QBM with the exact gradient and with the bound gradient; transverse-field scan of the trained XOR model | Table 5-classical, Fig. 5-classical | 100 s |
| `quantum_data.py` | Hamiltonian learning from a Gibbs state inside the family; the XXZ target learned by three families of growing operator content | Table 5-hamlearn, Fig. 5-quantumdata | 3 s |
| `vqt.py` | The variational quantum thermalizer: product or full latent distribution followed by the chapter-4 circuit, free-energy gradient (analytic + parameter shift), transverse-field Ising chain at beta = 1 and 3 | Table 5-vqt, Fig. 5-vqt | 15 s |

Conventions: qubit 0 is the leftmost tensor factor; the visible qubits
come first and the hidden ones last; the Hamiltonian is
`H = -sum_mu theta_mu Lambda_mu` with Pauli strings `Lambda_mu` given as
dicts `{qubit: "X"/"Y"/"Z"}`; the visible bit `v_i = 0, 1` corresponds to
the `Z_i` eigenvalue `1 - 2 v_i`; bit strings are ordered by their integer
value with bit 0 most significant.  `classical_data.py` is the slow one
(the bars-and-stripes model has seven qubits and the exact gradient needs
one eigendecomposition per parameter and step).
