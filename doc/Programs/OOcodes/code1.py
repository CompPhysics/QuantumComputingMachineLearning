
import numpy as np



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

        super().__init__((1/np.sqrt(2)) * [[1, 1],

                                           [1, -1]], [target])

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

            diff = 2 ** (n - 1 - target)        # index difference when flipping target qubit bit

            step = diff * 2                    # step to next pair

            new_state = self.state.copy()

            # Iterate over pairs of amplitudes where target qubit is 0 vs 1

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

            mask_p = 2 ** (n - 1 - p)  # binary mask for qubit p

            mask_q = 2 ** (n - 1 - q)  # binary mask for qubit q

            new_state = self.state.copy()

            # Loop over all basis indices where p and q bits are 0

            for base in range(len(self.state)):

                if (base & mask_p) != 0 or (base & mask_q) != 0:

                    continue  # skip if base index already has p or q bit = 1

                # Construct indices for basis states |p_bit q_bit⟩ = |00>,|01>,|10>,|11>

                idx00 = base

                idx01 = base + mask_q       # q bit = 1

                idx10 = base + mask_p       # p bit = 1

                idx11 = base + mask_p + mask_q  # p and q = 1

                # Get current amplitudes for these four basis states

                a00, a01 = self.state[idx00], self.state[idx01]

                a10, a11 = self.state[idx10], self.state[idx11]

                # Apply 4x4 gate matrix to [a00, a01, a10, a11]^T

                result = gate.matrix @ np.array([a00, a01, a10, a11], dtype=np.complex128)

                # Update the new state vector with transformed amplitudes

                new_state[idx00], new_state[idx01] = result[0], result[1]

                new_state[idx10], new_state[idx11] = result[2], result[3]

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

            beta = self.state[1] if len(self.state) > 1 else 0

            # Compute Bloch sphere coordinates (x,y,z) from state α|0> + β|1>

            x = 2 * np.real(alpha * np.conj(beta))

            y = 2 * np.imag(alpha * np.conj(beta))

            z = np.abs(alpha)**2 - np.abs(beta)**2

            # Plot a 3D Bloch sphere with the state vector

            import matplotlib.pyplot as plt

            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(5,5))

            ax = fig.add_subplot(111, projection='3d')

            # Draw sphere wireframe

            u = np.linspace(0, 2*np.pi, 36)

            v = np.linspace(0, np.pi, 18)

            xs = np.outer(np.cos(u), np.sin(v))

            ys = np.outer(np.sin(u), np.sin(v))

            zs = np.outer(np.ones_like(u), np.cos(v))

            ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.3)

            # Draw coordinate axes

            ax.quiver(0,0,0, 1,0,0, color='red', arrow_length_ratio=0.1)

            ax.quiver(0,0,0, 0,1,0, color='green', arrow_length_ratio=0.1)

            ax.quiver(0,0,0, 0,0,1, color='blue', arrow_length_ratio=0.1)

            ax.text(1.1, 0, 0, 'X', color='red'); ax.text(0, 1.1, 0, 'Y', color='green'); ax.text(0, 0, 1.1, 'Z', color='blue')

            # Plot state vector as an arrow

            ax.quiver(0,0,0, x, y, z, color='purple', arrow_length_ratio=0.2, linewidth=2)

            ax.set_box_aspect([1,1,1]); plt.axis('off')

            return fig

        else:

            # Bar chart of outcome probabilities for multi-qubit state

            import matplotlib.pyplot as plt

            probs = self.get_probabilities()

            fig = plt.figure(figsize=(6,4))

            ax = fig.add_subplot(111)

            num_states = len(probs)

            ax.bar(range(num_states), probs, color='teal')

            # Label each bar with the binary basis state

            labels = [format(i, f'0{self.num_qubits}b') for i in range(num_states)]

            ax.set_xticks(range(num_states)); ax.set_xticklabels(labels)

            ax.set_xlabel('Basis state'); ax.set_ylabel('Probability')

            ax.set_title('State probabilities')

            plt.tight_layout()

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

