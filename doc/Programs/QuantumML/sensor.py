from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
import numpy as np
import matplotlib.pyplot as plt

def create_quantum_circuit(theta):
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)  # Hadamard gate
    circuit.ry(theta, 0)  # RY gate with parameter theta
    circuit.measure(0, 0)
    return circuit

def simulate_quantum_circuit(circuit, shots=1024):
    simulator = Aer.get_backend('qasm_simulator')
    compiled_circuit = transpile(circuit, simulator)
    result = execute(compiled_circuit, simulator, shots=shots).result()
    counts = result.get_counts(compiled_circuit)
    return counts

def parameter_shift(theta, target, shots=1024, delta=0.001, learning_rate=0.01):
    forward_circuit = create_quantum_circuit(theta + delta)
    backward_circuit = create_quantum_circuit(theta - delta)

    forward_counts = simulate_quantum_circuit(forward_circuit, shots)
    backward_counts = simulate_quantum_circuit(backward_circuit, shots)

    forward_expectation = forward_counts.get('0', 0) / shots
    backward_expectation = backward_counts.get('0', 0) / shots

    gradient = (forward_expectation - backward_expectation) / (2 * delta)

    return theta - learning_rate * gradient

def train_quantum_sensor(target, epochs=200, convergence_threshold=0.01):
    theta = -1.0
    # Lists to store data for final plot
    epochs_list = []
    cost_values = []

    for epoch in range(epochs):
        circuit = create_quantum_circuit(theta)
        counts = simulate_quantum_circuit(circuit)

        # Evaluate the cost function (simple example)
        cost = np.abs(counts.get('0', 0) / 1024 - target)

        # Update theta using the parameter-shift rule
        theta = parameter_shift(theta, convergence_threshold)  # Pass convergence threshold here

        # Append data for final plot
        epochs_list.append(epoch + 1)
        cost_values.append(cost)

        # Print numbers
        print(f"Epoch {epoch + 1}/{epochs}, Cost: {cost}, Theta: {theta}")

        # Check for convergence
        if cost < convergence_threshold:
            print(f"Converged at epoch {epoch + 1} with cost {cost}")
            break

    # Final plot
    plt.plot(epochs_list, cost_values, marker='o', linestyle='-', color='b')
    plt.title('Cost Function Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

    return theta
# Example usage
target_value = 0.9  # Replace with your target value
trained_theta = train_quantum_sensor(target_value, convergence_threshold=0.0005
)
print(f"Trained Theta: {trained_theta}")
