import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and binarize MNIST
def load_binarized_mnist(n_samples=1000):
    print("Downloading MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data[:n_samples].astype(np.float32) / 255.0
    X = (X > 0.5).astype(np.float32)
    return X

# Quantum device and number of hidden qubits
n_visible = 8  # We will use only 8 pixels for this toy example
n_hidden = 4
dev = qml.device("default.qubit", wires=n_hidden)

# Define quantum circuit as hidden layer
def qrbm_circuit(v, weights):
    for i in range(n_hidden):
        qml.Hadamard(wires=i)
        qml.RZ(weights[i], wires=i)
    for i in range(n_hidden - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_hidden)]

# QRBM class
class QRBM:
    def __init__(self, n_visible, n_hidden, lr=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = np.random.uniform(0, 2 * np.pi, n_hidden, requires_grad=True)
        self.visible_bias = np.zeros(n_visible, requires_grad=False)
        self.optimizer = qml.GradientDescentOptimizer(stepsize=lr)

    def reconstruct(self, v):
        # Use quantum circuit to sample hidden state
        h_exp = hidden_layer(v, self.weights)
        v_recon = np.tanh(np.dot(h_exp, np.random.randn(self.n_hidden, self.n_visible)) + self.visible_bias)
        return (v_recon > 0.5).astype(np.float32)

    def cost(self, v):
        recon = self.reconstruct(v)
        return np.mean((v - recon) ** 2)

    def train(self, data, epochs=10):
        for epoch in range(epochs):
            total_cost = 0
            for v in data:
                v = v[:self.n_visible]
                self.weights, cost_val = self.optimizer.step_and_cost(lambda w: self.cost(v), self.weights)
                total_cost += cost_val
            print(f"Epoch {epoch+1}: Cost = {total_cost / len(data):.4f}")
# Load and reduce MNIST
X = load_binarized_mnist(n_samples=500)
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

# Train QRBM
qrbm = QRBM(n_visible=n_visible, n_hidden=n_hidden, lr=0.3)
# Convert X_train to a NumPy array before training
qrbm.train(X_train.to_numpy(), epochs=10) # Converting X_train to numpy array

# Visualize a reconstruction
def show_reconstruction(original, reconstructed):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original[:n_visible].reshape(1, -1), cmap="gray", aspect='auto')
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed.reshape(1, -1), cmap="gray", aspect='auto')
    axes[1].set_title("Reconstruction")
    plt.show()

# Access the first row using .iloc to avoid the KeyError
sample = X_test.iloc[0].values # Access the first row and convert to NumPy array
reconstruction = qrbm.reconstruct(sample[:n_visible])
show_reconstruction(sample, reconstruction)
