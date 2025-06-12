import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Ensure this is imported if visualize_state is used

class Qubit:
    def __init__(self, index):
        self.index = index

class Gate:
    """Base class for quantum gates."""
    def __init__(self, matrix, targets):
        self.matrix = np.array(matrix, dtype=np.complex128)
        # Convert Qubit objects to indices if needed
        self.targets = [(t.index if isinstance(t, Qubit) else t) for t in targets]
        self.num_targets = len(self.targets)
        self.name = "CustomGate"

    def __repr__(self):
        return f"{self.name}(targets={self.targets})"

# One-qubit gate subclasses:
class IGate(Gate):
    def __init__(self, target):
        super().__init__([[1, 0],
                          [0, 1]], [target])
        self.name = "I"

class XGate(Gate):
    def __init__(self, target):
        super().__init__([[0, 1],
                          [1, 0]], [target])
        self.name = "X"

class YGate(Gate):
    def __init__(self, target):
        super().__init__([[0, -1j],
                          [1j, 0]], [target])
        self.name = "Y"

class ZGate(Gate):
    def __init__(self, target):
        super().__init__([[1, 0],
                          [0, -1]], [target])
        self.name = "Z"

class HGate(Gate):
    def __init__(self, target):
        # Convert the list to a NumPy array before multiplication
        hadamard_matrix = np.array([[1, 1],
                                           [1, -1]], dtype=np.complex128)
        scaled_matrix = (1 / np.sqrt(2)) * hadamard_matrix
        super().__init__(scaled_matrix, [target])
        self.name = "H"

class SGate(Gate):
    def __init__(self, target):
        super().__init__([[1, 0],
                          [0, 1j]], [target])
        self.name = "S"

class TGate(Gate):
    def __init__(self, target):
        super().__init__([[1, 0],
                          [0, np.exp(1j*np.pi/4)]], [target])
        self.name = "T"

class RXGate(Gate):
    def __init__(self, target, theta):
        matrix = [[np.cos(theta/2), -1j*np.sin(theta/2)],
                  [-1j*np.sin(theta/2), np.cos(theta/2)]]
        super().__init__(matrix, [target])
        self.name = f"R_x({theta})"

class RYGate(Gate):
    def __init__(self, target, theta):
        matrix = [[np.cos(theta/2), -np.sin(theta/2)],
                  [np.sin(theta/2),  np.cos(theta/2)]]
        super().__init__(matrix, [target])
        self.name = f"R_y({theta})"

class RZGate(Gate):
    def __init__(self, target, theta):
        matrix = [[np.exp(-1j*theta/2), 0],
                  [0, np.exp(1j*theta/2)]]
        super().__init__(matrix, [target])
        self.name = f"R_z({theta})"

# Two-qubit gate subclasses:
class CNOTGate(Gate):
    """Controlled-NOT gate: flips target if control is 1."""
    def __init__(self, control, target):
        matrix = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]]
        super().__init__(matrix, [control, target])
        self.name = "CNOT"

class CZGate(Gate):
    """Controlled-Z gate: phase flip on |11>."""
    def __init__(self, control, target):
        matrix = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, -1]]
        super().__init__(matrix, [control, target])
        self.name = "CZ"

class SWAPGate(Gate):
    """SWAP gate: exchange two qubit states."""
    def __init__(self, qubit1, qubit2):
        matrix = [[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]]
        super().__init__(matrix, [qubit1, qubit2])
        self.name = "SWAP"

class TwoQubitGate(Gate):
    """Generic two-qubit gate defined by a 4x4 unitary matrix."""
    def __init__(self, matrix, qubit1, qubit2):
        super().__init__(matrix, [qubit1, qubit2])
        self.name = "Custom2QGate"

class Circuit:
    def __init__(self, num_qubits):
        # Initialize qubits and state vector |00...0>
        self.qubits = [Qubit(i) for i in range(num_qubits)]
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # start in |0...0>
        self.gates = []

    def add_gate(self, gate):
        # Ensure gate targets are valid for this circuit
        for t in gate.targets:
            if t < 0 or t >= self.num_qubits:
                raise ValueError(f"Qubit index {t} out of range for {self.num_qubits} qubits.")
        self.gates.append(gate)

    def apply_gate(self, gate):
        """Apply a single gate's unitary to the current state vector."""
        if gate.num_targets == 1:
            # One-qubit gate
            target = gate.targets[0]
            n = self.num_qubits
            # Calculate indices for pairs (target_qubit=0, target_qubit=1)
            diff = 2 ** (n - 1 - target)
            step = diff * 2
            new_state = self.state.copy()

            # Iterate over pairs of amplitudes where target qubit is 0 vs 1
            # This loop structure correctly handles applying a single-qubit gate
            # across the larger state space.
            for i in range(0, len(self.state), step):
                for j in range(diff):
                    idx0 = i + j             # index where target qubit is 0
                    idx1 = idx0 + diff       # index where target qubit is 1
                    a0, a1 = self.state[idx0], self.state[idx1]

                    # Apply 2x2 matrix U to [a0, a1]
                    new_state[idx0] = gate.matrix[0][0]*a0 + gate.matrix[0][1]*a1
                    new_state[idx1] = gate.matrix[1][0]*a0 + gate.matrix[1][1]*a1
            self.state = new_state

        elif gate.num_targets == 2:
            # Two-qubit gate
            p, q = gate.targets  # the two qubit indices
            n = self.num_qubits
            # Ensure p is the control and q is the target for CNOT/CZ logic if needed,
            # but for a generic 4x4 matrix application, the order in self.targets matters.
            # The matrix is assumed to act on qubits p and q in that order.

            # Determine the positions of the qubits in the state vector indexing
            # Indexing is typically big-endian (most significant bit first).
            # If qubit 0 is the left-most bit, its mask is 2^(n-1), qubit 1 is 2^(n-2), etc.
            mask_p = 2 ** (n - 1 - p)
            mask_q = 2 ** (n - 1 - q)

            new_state = self.state.copy()

            # Iterate through all possible lower bits combinations (excluding p and q)
            # A more efficient way is to iterate directly through the 2x2 sub-blocks
            # but this index-based approach is conceptually clearer for the 4x4 application.
            # We can iterate over all indices and only process those where the bits
            # corresponding to p and q are both 0 in the 'base' index.
            for base in range(len(self.state)):
                # Check if the p-th and q-th bits are both 0 in the current index `base`.
                # This finds the starting index of each 4x4 block in the state vector.
                if (base & mask_p) != 0 or (base & mask_q) != 0:
                    continue # Skip if this is not a 'base' index (where p and q bits are 0)

                # Construct the four indices corresponding to the basis states |p_bit q_bit⟩
                # based on the masks. These must correspond to the order expected by the 4x4 matrix.
                # The standard order is |00>, |01>, |10>, |11>.
                # Assuming the gate matrix is ordered for (qubit p, qubit q):
                # |00> corresponds to index `base`
                # |01> corresponds to index where p bit is 0 and q bit is 1
                # |10> corresponds to index where p bit is 1 and q bit is 0
                # |11> corresponds to index where p bit is 1 and q bit is 1
                # The indices are constructed by adding the masks.
                idx00 = base
                idx01 = base + mask_q
                idx10 = base + mask_p
                idx11 = base + mask_p + mask_q

                # Get current amplitudes for these four basis states
                a00, a01, a10, a11 = self.state[idx00], self.state[idx01], self.state[idx10], self.state[idx11]

                # Apply the 4x4 gate matrix to the vector of these four amplitudes
                result = gate.matrix @ np.array([a00, a01, a10, a11], dtype=np.complex128)

                # Update the new state vector with the transformed amplitudes
                new_state[idx00], new_state[idx01], new_state[idx10], new_state[idx11] = result[0], result[1], result[2], result[3]

            self.state = new_state

        else:
            raise ValueError("Gate with unsupported number of targets.")


    def run(self):
        """Apply all gates in sequence to evolve the quantum state."""
        for gate in self.gates:
            self.apply_gate(gate)
        return self.state

    def reset(self):
        """Reset the circuit state back to |00...0⟩."""
        self.state[:] = 0
        self.state[0] = 1.0

    def get_statevector(self):
        return self.state.copy()

    def get_probabilities(self):
        """Return a list of probabilities for each computational basis state."""
        return np.abs(self.state)**2

    def visualize_state(self):
        """Basic visualization of the current state: Bloch sphere for 1 qubit, or probabilities for multiple qubits."""
        if self.num_qubits == 1:
            # Bloch sphere visualization for single qubit state
            alpha = self.state[0]
            beta = self.state[1] if len(self.state) > 1 else 0 # Should always be len 2 for 1 qubit

            # Ensure the state is normalized for Bloch sphere calculation
            norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
            if norm > 1e-9: # Avoid division by zero or very small numbers
                alpha /= norm
                beta /= norm
            else:
                 # Handle the case of a zero state vector (shouldn't happen in unitary evolution)
                 # Or just plot at the origin if the state is somehow zero
                 x, y, z = 0, 0, 0


            # Compute Bloch sphere coordinates (x,y,z) from state α|0> + β|1>
            # x = Tr(rho * sigma_x) where rho = |psi><psi| and sigma_x is Pauli X
            # For psi = alpha|0> + beta|1>, rho = |alpha|^2 |0><0| + alpha*conj(beta)|0><1| + conj(alpha)*beta|1><0| + |beta|^2|1><1|
            # sigma_x = |0><1| + |1><0|
            # Tr(rho * sigma_x) = alpha*conj(beta) * Tr(|0><1|*|0><1|) + alpha*conj(beta) * Tr(|0><1|*|1><0|) + ...
            # Tr(|a><b|*|c><d|) = <b|c><d|a>
            # Tr(|0><1|*|1><0|) = <1|1><0|0> = 1*1 = 1
            # Tr(|1><0|*|0><1|) = <0|0><1|1> = 1*1 = 1
            # Tr(|0><1|*|0><1|) = <1|0><1|0> = 0
            # Tr(|1><0|*|1><0|) = <0|1><0|1> = 0
            # x = alpha*conj(beta) + conj(alpha)*beta = 2 * real(alpha * conj(beta))
            # Similarly:
            # y = Tr(rho * sigma_y) = alpha*conj(beta)*(-i) + conj(alpha)*beta*(i) = -i(alpha*conj(beta) - conj(alpha)*beta) = 2 * imag(alpha * conj(beta))
            # z = Tr(rho * sigma_z) = |alpha|^2 * 1 + |beta|^2 * (-1) = |alpha|^2 - |beta|^2

            x = 2 * np.real(alpha * np.conj(beta))
            y = 2 * np.imag(alpha * np.conj(beta))
            z = np.abs(alpha)**2 - np.abs(beta)**2

            # Plot a 3D Bloch sphere with the state vector
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection='3d')

            # Draw sphere wireframe
            u = np.linspace(0, 2*np.pi, 36)
            v = np.linspace(0, np.pi, 18)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.3)

            # Draw coordinate axes (Scaled slightly for better visualization)
            ax.quiver(0,0,0, 1.2,0,0, color='red', arrow_length_ratio=0.05) # X-axis
            ax.quiver(0,0,0, 0,1.2,0, color='green', arrow_length_ratio=0.05) # Y-axis
            ax.quiver(0,0,0, 0,0,1.2, color='blue', arrow_length_ratio=0.05) # Z-axis
            ax.text(1.3, 0, 0, 'X', color='red', fontsize=12)
            ax.text(0, 1.3, 0, 'Y', color='green', fontsize=12)
            ax.text(0, 0, 1.3, 'Z', color='blue', fontsize=12)


            # Plot state vector as an arrow
            ax.quiver(0,0,0, x, y, z, color='purple', arrow_length_ratio=0.15, linewidth=2)

            # Set plot limits and aspect ratio
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])
            ax.set_box_aspect([1,1,1]) # Equal aspect ratio for x, y, z

            plt.axis('off') # Hide axes labels and ticks
            ax.set_title('Bloch Sphere Representation')

            return fig

        else:
            # Bar chart of outcome probabilities for multi-qubit state
            probs = self.get_probabilities()
            fig = plt.figure(figsize=(8,5)) # Slightly larger figure
            ax = fig.add_subplot(111)

            num_states = len(probs)
            # Generate labels for the x-axis (binary strings)
            labels = [format(i, f'0{self.num_qubits}b') for i in range(num_states)]

            # Use np.arange for positions and labels for ticks
            x_positions = np.arange(num_states)
            ax.bar(x_positions, probs, color='teal', width=0.8) # Adjust width if needed

            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=45, ha='right') # Rotate labels for readability

            ax.set_xlabel('Basis state'); ax.set_ylabel('Probability')
            ax.set_title('State probabilities')
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            return fig

# Single-qubit circuit: start in |0>, apply H gate

circuit1 = Circuit(1)

circuit1.add_gate(HGate(0))

circuit1.run()

print("Final state vector:", circuit1.get_statevector())

# Visualize on Bloch sphere

circuit1.visualize_state()


# Two-qubit circuit: create a Bell state (|00> + |11>)/√2

circuit2 = Circuit(2)

circuit2.add_gate(HGate(0))       # Hadamard on qubit 0

circuit2.add_gate(CNOTGate(0, 1)) # CNOT with control 0, target 1

circuit2.run()

print("Final state vector:", circuit2.get_statevector())

# Visualize probabilities of each basis state

circuit2.visualize_state()

