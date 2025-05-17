import numpy as np
from scipy.linalg import expm

# Define constants
ħ = 1  # Planck's constant (for simplicity, set ħ = 1)

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Function to create an entangled Bell state
def create_bell_state():
    return (1/np.sqrt(2)) * (np.kron(np.array([[1], [0]]), np.array([[1], [0]])) + np.kron(np.array([[0], [1]]), np.array([[0], [1]])))

# Function to create the Hamiltonian for a magnetic field
def hamiltonian(B, t):
    # B: magnetic field (3D vector)
    # Heaviside-like treatment of the magnetic field
    
    H = -ħ * (B[0] * X + B[1] * Y + B[2] * Z)
    return H

# Function to evolve the state
def evolve_state(state, H, time):
    U = expm(-1j * H * time)
    return U @ state

# Function to measure the state and project it to the standard basis
def measure(state):
    probabilities = np.abs(state.flatten())**2
    outcome = np.random.choice(range(len(probabilities)), p=probabilities)
    return outcome, state.flatten()[outcome]

# Main program
if __name__ == "__main__":
    # Create initial entangled state
    initial_state = create_bell_state()
    print("Initial State:\n", initial_state)

    # Time-dependent magnetic field (example: oscillating field)
    time_points = np.linspace(0, 10, 100)
    B = np.array([np.cos(time_points), np.sin(time_points), np.zeros_like(time_points)])  # Magnetic field oscillating in XY-plane

    # Simulate evolution
    for t in time_points:
        H = hamiltonian(B[:, int(t)], t)  # Hamiltonian for current time point
        evolved_state = evolve_state(initial_state, H, 0.1)  # Evolve for a small time step

    # Measure the final state
    outcome, final_state = measure(evolved_state)
    print("Measurement Outcome:", outcome)
    print("Final State After Measurement:\n", final_state)
