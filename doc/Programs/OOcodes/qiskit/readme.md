
# Modular Quantum Simulator (Qiskit Version)

This is a fully modular quantum simulator package implemented with Qiskit.

## Modules

- `core/gates.py` — gate factory (wrapper around Qiskit gates)
- `core/circuit.py` — modular circuit builder
- `states/bell_states.py` — Bell state generators
- `examples/bell_test.py` — Bell state experiment

## Features

- Fully based on Qiskit's QuantumCircuit
- Supports any number of qubits
- Native statevector simulation and measurement
- Measurement histograms
- Mirrors NumPy version structure for easy comparison

## Dependencies

- Qiskit
- Matplotlib