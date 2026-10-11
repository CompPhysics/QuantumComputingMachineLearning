# BookPrograms/common — modules shared by every chapter of the QML book

| Module | Contents |
|---|---|
| `gates.py` | The elementary gates as matrices (`X, Y, Z, H, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP, controlled`). Copied from the quantum-computing book, `doc/Book/BookPrograms/common/gates.py`. |
| `statevector.py` | `StateVector`: an n-qubit state-vector simulator (apply gates, probabilities, marginals, measurement, sampling). Copied from the quantum-computing book. Qubit 0 is the leftmost tensor factor. |
| `datasets.py` | Labelled toy data with labels ±1, NumPy only: `two_blobs`, `two_moons`, `two_circles`, `xor`, `iris` (from `iris.csv`), `train_test_split`, `Scaler` (standard or into `[0, pi]`), `accuracy`. |
| `iris.csv` | Fisher's Iris data (150 rows: sepal length/width, petal length/width, species 0/1/2). |
| `neuralnet.py` | `MLP`: a small fully connected network in NumPy (tanh units, backpropagation, Adam), the classical reference model of chapter 4 |
| `optimize.py` | A small BFGS minimiser with backtracking line search (copy of the QC book's, with an optional loss history), the `Adam` optimiser and `gradient_descent`; used by the QAOA class and the quantum neural networks. |
| `bookplot.py` | Figure directory and style; `savefig(fig, chapter, name)` writes PDF + PNG to `BookFigures/chapterN/`; `scatter_classes`, `decision_regions`. |

Every chapter program starts with the bootstrap that puts all `BookPrograms/*/` folders on `sys.path`, so these modules are imported by bare name (`from statevector import StateVector`).
