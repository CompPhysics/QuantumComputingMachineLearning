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
        # Original magnetic field B is still a 3x100 array representing Bx, By, Bz over time
        self.B = np.array([np.cos(self.time_points), np.sin(self.time_points), np.zeros_like(self.time_points)])
        self.estimated_Bx = None
        self.estimated_By = None
        self.estimated_Bz = None

    @staticmethod
    def create_bell_state():
        """Create an entangled Bell state."""
        # Returns a 4x1 column vector
        return (1/np.sqrt(2)) * (np.kron(np.array([[1], [0]]), np.array([[1], [0]])) +
                                     np.kron(np.array([[0], [1]]), np.array([[0], [1]])))

    @staticmethod
    def single_qubit_hamiltonian(B):
        """Create the Hamiltonian for a single qubit in a magnetic field B."""
        ħ = 1  # Planck's constant
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        return -ħ * (B[0] * X + B[1] * Y + B[2] * Z)

    @staticmethod
    def two_qubit_hamiltonian(B):
        """Create the Hamiltonian for a two-qubit system."""
        # Identity matrix for a single qubit
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        # Single qubit Hamiltonian
        H_single = QuantumSensor.single_qubit_hamiltonian(B)

        # Total Hamiltonian is the sum of the Hamiltonians on each qubit (assuming no interaction)
        # H_total = H_single (on qubit 1) + H_single (on qubit 2)
        # In tensor product notation: H_total = H_single \otimes I + I \otimes H_single
        return np.kron(H_single, I) + np.kron(I, H_single)


    @staticmethod
    def evolve_state(state, H, time):
        """Evolve the state using the unitary operator."""
        # H should now be a 4x4 matrix
        U = expm(-1j * H * time)
        # state is a 4x1 vector
        # U @ state will result in a 4x1 vector
        return U @ state

    def measure(self, state):
        """Measure the state in the standard basis."""
        probabilities = np.abs(state.flatten())**2
        # Ensure probabilities sum to 1 to avoid issues with floating point inaccuracies
        probabilities /= np.sum(probabilities)
        outcome = np.random.choice(range(len(probabilities)), p=probabilities)
        return outcome, state.flatten()[outcome]


    def simulate(self):
        """Run the simulation for a given number of shots."""
        for shot in range(self.shots):
            evolved_state = self.state
            for t in range(self.time_steps):
                # Use the two-qubit Hamiltonian
                H = self.two_qubit_hamiltonian(self.B[:, t])
                # Evolve the 4-dimensional state with the 4x4 Hamiltonian
                evolved_state = self.evolve_state(evolved_state, H, 0.1)
            
            # Measure the final state
            outcome, _ = self.measure(evolved_state)
            self.measurement_results[shot] = outcome

    def estimate_magnetic_field(self):
        """Estimate the magnetic field from measurement outcomes."""
        
        # Take the average of the measurement results to infer the magnetic field
        n_0 = np.sum(self.measurement_results == 0)  # Number of outcomes corresponding to |00>
        n_1 = np.sum(self.measurement_results == 1)  # Number of outcomes corresponding to |01>
        n_2 = np.sum(self.measurement_results == 2)  # Number of outcomes corresponding to |10>
        n_3 = np.sum(self.measurement_results == 3)  # Number of outcomes corresponding to |11>
        
        counts = np.array([n_0, n_1, n_2, n_3])
        probabilities = counts / self.shots

        # Using probabilities to estimate the components of the magnetic field
        # These simplified estimations are likely based on the expected probabilities
        # of measuring |01> and |10> states for a Bell state evolving under Bx and By fields.
        # A more rigorous estimation might involve a maximum likelihood estimation or similar.
        self.estimated_Bx = 2 * (probabilities[1] - probabilities[2])  # Simplified estimation
        self.estimated_By = 2 * (probabilities[3] - probabilities[0])  # Simplified estimation
        self.estimated_Bz = 0  # Assuming Bz doesn't cause |00> or |11> to change in this simplified model


    def plot_results(self):
        """Plot the histogram of measurement outcomes."""
        plt.figure(figsize=(10, 6))
        # Use align='left' and add 0.5 to bins for proper alignment with xticks
        plt.hist(self.measurement_results, bins=np.arange(5) - 0.5, density=True, alpha=0.7, color='blue', edgecolor='black', align='left')
        plt.xticks(range(4))
        plt.xlabel('Measurement Outcome')
        plt.ylabel('Probability')
        plt.title(f'Histogram of Measurement Outcomes over {self.shots} Shots')
        plt.grid()
        plt.show()

    def plot_estimated_field(self):
        """Plot the estimated magnetic field against the actual magnetic field."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.time_points, self.B[0], 'r-', label='Actual Bx')
        # Estimated Bx is a single value, plot as a horizontal line
        plt.axhline(self.estimated_Bx, color='blue', linestyle='--', label=f'Estimated Bx = {self.estimated_Bx:.2f}')
        plt.title('Estimated vs Actual Magnetic Field')
        plt.ylabel('Bx')
        plt.legend()
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(self.time_points, self.B[1], 'g-', label='Actual By')
        # Estimated By is a single value, plot as a horizontal line
        plt.axhline(self.estimated_By, color='blue', linestyle='--', label=f'Estimated By = {self.estimated_By:.2f}')
        plt.ylabel('By')
        plt.legend()
        plt.grid()

        plt.subplot(3, 1, 3)
        # Estimated Bz is fixed at 0 in this model, plot as a horizontal line
        plt.axhline(self.estimated_Bz, color='blue', linestyle='--', label='Estimated Bz = 0')
        plt.ylabel('Bz')
        plt.xlabel('Time')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    sensor = QuantumSensor(shots=1000, time_steps=100)
    sensor.simulate()
    sensor.plot_results()
    sensor.estimate_magnetic_field()
    
    estimated_Bx, estimated_By, estimated_Bz = sensor.estimated_Bx, sensor.estimated_By, sensor.estimated_Bz
    print(f"Estimated Magnetic Field: Bx = {estimated_Bx:.2f}, By = {estimated_By:.2f}, Bz = {estimated_Bz:.2f}")
    
    sensor.plot_estimated_field()

