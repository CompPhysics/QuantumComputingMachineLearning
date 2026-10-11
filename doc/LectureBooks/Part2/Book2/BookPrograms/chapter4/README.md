# Programs for chapter 4, Quantum Neural Networks

All programs are NumPy + matplotlib only and import the shared modules of
`../common/` (`gates.py`, `statevector.py`, `optimize.py` with BFGS and
Adam, `neuralnet.py`, `datasets.py`, `bookplot.py`); `portfolio_qnn.py`
also imports `../chapter3/portfolio.py`.  Figures go to
`../../BookFigures/chapter4/` (PDF + PNG).  Every number quoted in
`chapter4.tex` is printed by one of them.

| Program | What it computes | Output | Runtime |
|---|---|---|---|
| `qnn.py` | The `QNN` class: angle encoding with data re-uploading, `R_y`/`R_z` layers with a CNOT ladder, `<Z_{n-1}>` output, batched over inputs and parameter sets; parameter-shift gradient, state derivatives, Fourier coefficients; `trig_weights` (general shift rule) | self-test, Fig. 4-fourier | 2 s |
| `training.py` | `QNNRegressor`, `QNNClassifier`: Adam training on the square loss with an output scale and offset | self-test | 1 s |
| `qfi.py` | Quantum Fisher information from state derivatives and from overlaps of shifted states; natural gradient; ground state of the transverse-field Ising chain with GD / Adam / natural gradient | Table 4-qfirank, Fig. 4-natgrad | 11 s |
| `barren.py` | Gradient variance of random hardware-efficient circuits against width and depth, local and global cost | Table 4-barren, Fig. 4-barren | 65 s |
| `regression.py` | The noisy cubic with two-qubit QNNs of depth 1-6 against a 1-16-16-1 network | Table 4-regression, Fig. 4-regression | 17 s |
| `classifier.py` | The variational classifier on the four two-class problems of chapter 2, against the RBF SVM and the angle QSVM | Table 4-classifier, Fig. 4-classifier | 17 s |
| `portfolio_qnn.py` | Single-qubit re-uploading scoring circuit with soft top-K selection on the two portfolio problems of chapter 3 | numbers of Sec. 4.7.3 | 5 s |
| `qpinn.py` | `Functional` (linear functionals of the circuit output), `QPINN` (derivative functionals by the general shift rule and by the per-gate product rule), `residual_loss`, `ritz_loss`, `train` | Table 4-derivatives | 1 s |
| `poisson.py` | The 1D Poisson problem with hard Dirichlet conditions, strong form for L = 1, 2, 3 and Ritz form | Table 4-poisson, Fig. 4-poisson | 11 s |
| `timedependent.py` | The harmonic oscillator (hard initial conditions) and the wave equation in (x, t) with additive trial functions, L = 1 and 2 | Fig. 4-timedependent | 30 s |

Conventions: qubit 0 is the leftmost bit; `R_a(t) = exp(-i t sigma_a/2)`;
parameters are ordered layer, rotation axis (y then z), qubit; the
observable is Z on the last qubit (the end of the CNOT ladder) unless a
diagonal observable is passed.  The input derivatives of `QPINN` require
all gates carrying a coordinate to share the same scale (`scales[q, i]`).
