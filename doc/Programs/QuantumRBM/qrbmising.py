import numpy as np

# Define Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

# Kronecker product helper
def kron(*matrices):
    result = np.array([[1]])
    for m in matrices:
        result = np.kron(result, m)
    return result

# Rotation around Y axis
def Ry(theta):
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ])

# CNOT gate on qubits 0 (control) and 1 (target)
CNOT = np.array([
    [1, 0, 0, 0],  # |00⟩ -> |00⟩
    [0, 1, 0, 0],  # |01⟩ -> |01⟩
    [0, 0, 0, 1],  # |10⟩ -> |11⟩
    [0, 0, 1, 0],  # |11⟩ -> |10⟩
])

# Full variational circuit with entanglement
def variational_state(theta1, theta2):
    # Start in |00⟩
    state = np.array([1, 0, 0, 0], dtype=complex)
    
    # Apply Ry ⊗ Ry
    U = kron(Ry(theta1), Ry(theta2))
    state = U @ state
    
    # Apply CNOT
    state = CNOT @ state
    
    return state

# Hamiltonian (Ising interaction)
H = -kron(Z, Z)

# Compute expectation ⟨ψ|H|ψ⟩
def energy(theta1, theta2):
    psi = variational_state(theta1, theta2)
    return np.real(np.vdot(psi, H @ psi))

# Parameter shift rule: dE/dθ ≈ [E(θ + π/2) - E(θ - π/2)] / 2
def parameter_shift_grad(theta1, theta2):
    shift = np.pi / 2
    dtheta1 = 0.5 * (energy(theta1 + shift, theta2) - energy(theta1 - shift, theta2))
    dtheta2 = 0.5 * (energy(theta1, theta2 + shift) - energy(theta1, theta2 - shift))
    return dtheta1, dtheta2

# Training with gradient descent
theta1, theta2 = np.random.uniform(0, 2 * np.pi, 2)
learning_rate = 0.1

for step in range(100):
    E = energy(theta1, theta2)
    grad1, grad2 = parameter_shift_grad(theta1, theta2)
    
    theta1 -= learning_rate * grad1
    theta2 -= learning_rate * grad2
    
    if step % 10 == 0:
        print(f"Step {step:3d}: Energy = {E:.6f}, θ1 = {theta1:.4f}, θ2 = {theta2:.4f}")

print("\nFinal energy:", energy(theta1, theta2))
print("Final parameters: θ1 =", theta1, ", θ2 =", theta2)
