import numpy as np

def hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def apply_single_qubit_gate(state, gate, target, n_qubits):
    """Applies a single-qubit gate to the target qubit."""
    I = np.eye(2)
    ops = [I] * n_qubits
    ops[target] = gate
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U @ state

def apply_controlled_phase(state, control, target, theta, n_qubits):
    """Applies a controlled phase rotation gate."""
    size = 2 ** n_qubits
    U = np.eye(size, dtype=complex)
    for i in range(size):
        bin_str = format(i, f'0{n_qubits}b')
        if bin_str[n_qubits - 1 - control] == '1' and bin_str[n_qubits - 1 - target] == '1':
            U[i, i] *= np.exp(1j * theta)
    return U @ state

def swap_registers(state, n_qubits):
    """Swap qubit order (bit-reversal permutation)."""
    perm = [int(format(i, f'0{n_qubits}b')[::-1], 2) for i in range(2 ** n_qubits)]
    return state[perm]

def qft_step_by_step(state, n_qubits, inverse=False):
    """Performs (inverse) QFT using gate decomposition."""
    for target in range(n_qubits):
        idx = n_qubits - 1 - target  # Apply to qubits in reversed order
        # Controlled phase rotations
        for control_offset in range(1, n_qubits - target):
            control = n_qubits - 1 - (target + control_offset)
            angle = np.pi / (2 ** control_offset)
            if inverse:
                angle *= -1
            state = apply_controlled_phase(state, control, idx, angle, n_qubits)
        # Hadamard
        state = apply_single_qubit_gate(state, hadamard(), idx, n_qubits)
    # Swap qubits unless inverse and user wants to avoid it
    return swap_registers(state, n_qubits)

def measure_state(state, shots=1024):
    """Measure the quantum state multiple times to simulate outcomes."""
    probs = np.abs(state) ** 2
    outcomes = np.random.choice(len(probs), size=shots, p=probs)
    counts = {}
    for o in outcomes:
        b = format(o, f'0{int(np.log2(len(state)))}b')
        counts[b] = counts.get(b, 0) + 1
    return counts

# -------------------------
# Example: 3 qubits in |5⟩
# -------------------------
n_qubits = 3
N = 2 ** n_qubits
state = np.zeros(N, dtype=complex)
state[5] = 1.0  # Start in |5⟩ = |101⟩

# Apply QFT
qft_state = qft_step_by_step(state.copy(), n_qubits)

# Apply inverse QFT to check if we recover original
inv_qft_state = qft_step_by_step(qft_state.copy(), n_qubits, inverse=True)

# Measurement sampling from QFT output
measurement_results = measure_state(qft_state, shots=1024)

# -------------------------
# Output
# -------------------------
print("QFT amplitudes:")
for i, amp in enumerate(qft_state):
    print(f"|{i:03b}>: {amp:.4f}")

print("\nInverse QFT applied back (should recover |5⟩):")
for i, amp in enumerate(inv_qft_state):
    print(f"|{i:03b}>: {amp:.4f}")

print("\nMeasurement results from QFT output:")
for bitstring, count in sorted(measurement_results.items()):
    print(f"{bitstring}: {count}")

"""
Key Features
Inverse QFT (via negative controlled-phase angles and reversed order),
Measurement sampling (random draws from the final state’s probability distribution).

Forward and inverse QFT logic built in.
Uses exact Hadamard and controlled phase gates.
Simulates quantum measurement from the QFT result.
Bit-reversal swaps ensure correct output basis alignment.
"""
