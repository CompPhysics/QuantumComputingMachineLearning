import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class QuantumSensor:
    def __init__(self, shots=1000, time_steps=100):
        self.shots = shots
        self.time_steps = time_steps
        self.state = self.create_bell_state()
        self.measurement_results = np.zeros(shots)
        self.time_points = np.linspace(0, 10, self.time_steps)
        # B is a 3xN array where N is time_steps. B[:, t] gives the magnetic field vector at time t.
        self.B = np.array([np.cos(self.time_points), np.sin(self.time_points), np.zeros_like(self.time_points)])

    @staticmethod
    def create_bell_state():
        """Create an entangled Bell state (|00> + |11>)/sqrt(2)."""
        # Representing states in the computational basis |00>, |01>, |10>, |11>
        return (1/np.sqrt(2)) * (np.array([[1], [0], [0], [0]]) +
                                     np.array([[0], [0], [0], [1]]))

    @staticmethod
    def hamiltonian(B):
        """Create the Hamiltonian for the current magnetic field for a two-qubit system."""
        ħ = 1  # Planck's constant
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex) # 2x2 Identity matrix

        # Hamiltonian for two independent qubits interacting with the magnetic field
        # H = H1 + H2, where H1 = -ħ * B . sigma_1 and H2 = -ħ * B . sigma_2
        # In the two-qubit space (4x4 matrix):
        # H = -ħ * (Bx * (sigma_x tensor I) + By * (sigma_y tensor I) + Bz * (sigma_z tensor I) +
        #           Bx * (I tensor sigma_x) + By * (I tensor sigma_y) + Bz * (I tensor sigma_z))
        H1 = -ħ * (B[0] * np.kron(X, I) + B[1] * np.kron(Y, I) + B[2] * np.kron(Z, I))
        H2 = -ħ * (B[0] * np.kron(I, X) + B[1] * np.kron(I, Y) + B[2] * np.kron(I, Z))

        return H1 + H2

    @staticmethod
    def evolve_state(state, H, time):
        """Evolve the state using the unitary operator."""
        U = expm(-1j * H * time)
        return U @ state

    def measure(self, state):
        """Measure the state in the standard basis {|00>, |01>, |10>, |11>}."""
        # Ensure state is a column vector for probability calculation
        state_vector = state.flatten()
        probabilities = np.abs(state_vector)**2
        # Ensure probabilities sum to 1 (due to potential floating point inaccuracies)
        probabilities /= np.sum(probabilities)
        # Choose outcome based on probabilities. outcomes correspond to indices 0, 1, 2, 3
        outcome = np.random.choice(range(len(probabilities)), p=probabilities)
        # Return the outcome (index) and the corresponding state vector component
        return outcome, state_vector[outcome]


    def simulate(self):
        """Run the simulation for a given number of shots."""
        for shot in range(self.shots):
            evolved_state = self.state # Start each shot with the initial Bell state
            # The Hamiltonian is time-dependent, so we need to apply the evolution operator
            # for each small time step. The total evolution U(T) = U(dt_N) * ... * U(dt_1).
            # We are assuming a small constant time step dt = 0.1.
            for t in range(self.time_steps):
                # Get the magnetic field vector at the current time point
                current_B = self.B[:, t]
                # Calculate the Hamiltonian for the current magnetic field
                H = self.hamiltonian(current_B)
                # Evolve the state for this small time step
                evolved_state = self.evolve_state(evolved_state, H, 0.1)

            # Measure the final state after the total evolution time
            outcome, _ = self.measure(evolved_state)
            self.measurement_results[shot] = outcome

    def plot_results(self):
        """Plot the histogram of measurement outcomes."""
        plt.figure(figsize=(10, 6))
        # The possible outcomes for a two-qubit measurement are 0, 1, 2, 3
        plt.hist(self.measurement_results, bins=np.arange(5) - 0.5, density=True, rwidth=0.8)
        plt.xticks(range(4))
        plt.xlabel('Measurement Outcome (|00>, |01>, |10>, |11>)')
        plt.ylabel('Probability')
        plt.title(f'Histogram of Measurement Outcomes over {self.shots} Shots')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

# Main execution
if __name__ == "__main__":
    # Make sure expm is available, it's from scipy.linalg
    try:
        from scipy.linalg import expm
    except ImportError:
        print("Please install scipy: !pip install scipy")
        exit()

    sensor = QuantumSensor(shots=1000, time_steps=100)
    sensor.simulate()
    sensor.plot_results()

