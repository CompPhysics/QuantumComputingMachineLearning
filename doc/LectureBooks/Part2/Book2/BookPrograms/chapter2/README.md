# BookPrograms/chapter2 — Quantum Support Vector Machines

| Program | What it computes | Runtime |
|---|---|---|
| `kernels.py` | `Kernel` classes (`LinearKernel`, `PolynomialKernel`, `RBFKernel`, `TanhKernel`, `PrecomputedKernel`), the quadratic feature map, a numerical Mercer check; figures `kerneltrick`, `rbflandmarks`. | 3 s |
| `svm.py` | `SVM`: the dual problem solved by sequential minimal optimization for any kernel or a precomputed Gram matrix; bias, decision function, QP matrices, optional CVXOPT solution; figures `hardmargin`, `softmargin`, `moons` and the numbers quoted with them. | 10 s |
| `feature_maps.py` | `AngleEncoding`, `ZZFeatureMap`, `AmplitudeEncoding`, `TrainableEmbedding`; `QuantumKernel` with exact fidelities or shot-sampled estimates (adjoint or SWAP test) and the two test circuits on the simulator. | 1 s |
| `qsvm.py` | `QSVM` (quantum kernel + SMO, PSD projection of estimated kernels), `havlicek_dataset`; Tables 2-qsvmdata, 2-havlicek, 2-shots; figures `qsvmmoons`, `havlicek`, `shots`. | 30 s |
| `kernel_alignment.py` | `KernelAlignment`: kernel-target alignment, parameter-shift gradient (checked against finite differences), gradient ascent; figure `alignment`. | 40 s |

Import order: `kernels` ← `svm` ← `qsvm`; `feature_maps` ← `qsvm` ← `kernel_alignment`. Mirrors the chapter's closing `Programs for this chapter` list.
