# Quantum Machine Learning

*Morten Hjorth-Jensen*

This is the executable edition of the textbook. Every chapter is a Jupyter
notebook that carries the full text of the printed chapter, its equations
with the same numbers, its figures and tables, and the programs that
produced them. The code cells are not illustrations added afterwards:
every number quoted in the text is printed by one of them.

## How the book is organised

Each chapter takes a classical machine-learning model, derives it in
enough detail to implement it, replaces one of its components by a
quantum circuit, and runs both versions on the same data with the book's
own state-vector simulator.

| Chapter | Topic | Notebook |
|---|---|---|
| 1 | Introduction and motivation: what QML is, why quantum, the challenges, variational circuits reconsidered (trainability, classical surrogates, total cost; classical versus quantum data), the relation to the companion volume, data encodings | [chapter 1](chapter1/chapter1) |
| 2 | Quantum support vector machines: SVM theory, kernels, quantum feature maps and kernels, kernel estimation, QSVM, kernel-target alignment | [chapter 2](chapter2/chapter2) |
| 3 | The quantum approximate optimization algorithm: QUBO and Ising, the circuit, gradients, a double-quantum-dot register, portfolio selection, clustering | [chapter 3](chapter3/chapter3) |
| 4 | Quantum neural networks: the model and its Fourier series, shift rules, quantum Fisher information and the natural gradient, barren plateaus, regression and classification, physics-informed QNNs for differential equations | [chapter 4](chapter4/chapter4) |
| 5 | Quantum Boltzmann machines: energy-based models and the RBM, contrastive divergence, the QBM and its Duhamel gradient, the clamped-state bound, classical versus quantum data, Hamiltonian learning, the free-energy principle, thermofield doubles and the variational quantum thermalizer | [chapter 5](chapter5/chapter5) |
| 6 | Reinforcement learning and quantum reinforcement learning (placeholder: scope and intended structure) | [chapter 6](chapter6/chapter6) |
| 7 | Summary and perspectives: what the book has shown, the reassessment of variational circuits with the evidence in hand, a priority list for the next years, perspectives | [chapter 7](chapter7/chapter7) |

Citations appear in the text as bracketed keys such as [Havlicek2019]; the
[reference list](references) is labelled by the same keys.

## Running the code

The notebooks import the programs of the book rather than repeating them,
so they are meant to be run from a copy of the repository:

```bash
git clone https://github.com/QuantumComputingQuantumMachineLearning/QuantumComputing.git
cd QuantumComputing/doc/Book/BookQML/Notebooks
pip install numpy matplotlib jupyterlab
jupyter lab chapter2/chapter2.ipynb
```

The first code cell of every notebook puts `BookPrograms/chapterN/` and
`BookPrograms/common/` on the Python path. Each chapter ends with one cell
per program of its *Programs for this chapter* list, which runs the program
and shows the figures it wrote to `BookFigures/chapterN/`. The programs
need only `numpy` and `matplotlib`: the qubit simulator, the feature maps,
the quantum kernel and the support-vector machine solver are all written
out in the book's own classes, so that the mechanics stay visible. The
optional PennyLane, Qiskit and scikit-learn cells run when those packages
are installed and print a message when they are not. A whole chapter
executes in one to two minutes.

The pages shown here were executed when the book was built from the
LaTeX sources, and the outputs you see are the stored ones.
