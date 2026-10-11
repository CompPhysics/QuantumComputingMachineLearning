# Quantum Computing for Many-body Physics

**From non-relativistic quantum mechanics to quantum field theory**

*Alexei Bazavov, Zohreh Davoudi, Morten Hjorth-Jensen, Ryan LaRose, Dean Lee and Alessandro Roggero*

This is the executable edition of the textbook. Every chapter is a Jupyter
notebook that carries the full text of the printed chapter, its equations
with the same numbers, its figures and tables, and the programs that
produced them. The code cells are not illustrations added afterwards:
every number quoted in the text is printed by one of them.

## How the book is organised

The book follows four model Hamiltonians, the Lipkin model, the pairing
model, the Hubbard model and the Heisenberg model, from classical
many-body theory to quantum algorithms, at the same sizes and couplings
throughout, so that the tables of one chapter can be set beside those of
another.

| Chapter | Topic | Notebook |
|---|---|---|
| 1 | Introduction and motivation | [chapter 1](chapter1/chapter1) |
| 2 | Fundamentals of quantum computation | [chapter 2](chapter2/chapter2) |
| 3 | Quantum many-body physics primer: second quantization, FCI, Hartree–Fock, coupled cluster | [chapter 3](chapter3/chapter3) |
| 4 | Mapping many-body Hamiltonians to qubits | [chapter 4](chapter4/chapter4) |
| 5 | Hamiltonian simulation and time evolution | *in preparation* |
| 6 | Quantum Fourier transform, phase estimation and the HHL algorithm | [chapter 6](chapter6/chapter6) |
| 7 | Variational quantum algorithms: VQE, ADAPT-VQE, QAOA | [chapter 7](chapter7/chapter7) |
| 8 | Resolution refinement, the rodeo algorithm and filtering | [chapter 8](chapter8/chapter8) |
| 9 | Quantum error correction and fault tolerance | *in preparation* |
| 10 | A case study: toward simulating lattice gauge theories | *in preparation* |
| 11 | Conclusion and outlook | [chapter 11](chapter11/chapter11) |

Citations appear in the text as bracketed keys such as [Choi2021]; the
[reference list](references) is labelled by the same keys.

## Running the code

The notebooks import the programs of the book rather than repeating them,
so they are meant to be run from a copy of the repository:

```bash
git clone https://github.com/QuantumComputingQuantumMachineLearning/QuantumComputing.git
cd QuantumComputing/doc/Book/Notebooks
pip install numpy matplotlib jupyterlab
jupyter lab chapter8/chapter8.ipynb
```

The first code cell of every notebook puts `BookPrograms/chapterN/` and
`BookPrograms/common/` on the Python path. Each chapter ends with one cell
per program of its *Programs for this chapter* list, which runs the program
and shows the figures it wrote to `BookFigures/chapterN/`. The programs
need only `numpy` and `matplotlib`; the simulator, the Jordan–Wigner
mapping, phase estimation, the variational eigensolver and the rodeo
algorithm are all written out in the book's own classes, so that the
mechanics stay visible. A whole chapter executes in one to five minutes.

The pages shown here were executed when the book was built from the
LaTeX sources, and the outputs you see are the stored ones.
