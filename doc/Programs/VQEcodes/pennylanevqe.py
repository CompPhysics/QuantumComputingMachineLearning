import pennylane as qml
from pennylane import numpy as np

# Heisenberg coefficients
a1, a2, a3 = 1.0, 0.5, 0.8

# Quantum device (statevector backend)
dev = qml.device("default.qubit", wires=2)

# Ansatz circuit with 4 parameters
def ansatz(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)

# Measurement circuits
@qml.qnode(dev)
def measure_xx(params):
    ansatz(params)
    qml.Hadamard(0)
    qml.Hadamard(1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def measure_yy(params):
    ansatz(params)
    qml.adjoint(qml.S(0))
    qml.Hadamard(0)
    qml.adjoint(qml.S(1))
    qml.Hadamard(1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def measure_zz(params):
    ansatz(params)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Cost function = weighted sum of expectation values
def cost(params):
    return a1 * measure_xx(params) + a2 * measure_yy(params) + a3 * measure_zz(params)

# Initialize parameters
np.random.seed(42)
params = np.random.uniform(0, np.pi, 4, requires_grad=True)

# Optimization loop
opt = qml.AdamOptimizer(stepsize=0.2)
max_iter = 100

for i in range(max_iter):
    params, energy = opt.step_and_cost(cost, params)
    if i % 10 == 0:
        print(f"Iter {i:03d} | Energy: {energy:.6f}")

print("\nFinal VQE energy:", energy)


# Exact energy for comparison
# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

H_matrix = (
    a1 * np.kron(X, X)
  + a2 * np.kron(Y, Y)
  + a3 * np.kron(Z, Z)
)

eigvals = np.linalg.eigvalsh(H_matrix)
print("Exact ground state energy:", np.min(eigvals))
