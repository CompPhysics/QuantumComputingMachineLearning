import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Generate a simple binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
y = y * 2 - 1  # Convert to {-1, 1} for compatibility with Pauli measurements

# Step 2: Define the quantum device and circuit
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Step 3: Encode classical data into quantum state
def encode_data(x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)

# Step 4: Variational circuit (the "neural network")
def variational_layer(weights):
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)

@qml.qnode(dev)
def qnn_circuit(x, weights):
    encode_data(x)
    variational_layer(weights)
    return qml.expval(qml.PauliZ(0))

# Step 5: Define cost function
def cost(weights, X, y):
    predictions = [qnn_circuit(x, weights) for x in X]
    return np.mean((predictions - y)**2)

# Step 6: Train the QNN
weights = np.random.uniform(low=0, high=2 * np.pi, size=(2,), requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.2)

for epoch in range(30):
    weights = opt.step(lambda w: cost(w, X, y), weights)
    current_loss = cost(weights, X, y)
    print(f"Epoch {epoch+1:02d} | Loss: {current_loss:.4f}")

# Step 7: Prediction example
sample = X[0]
predicted = qnn_circuit(sample, weights)
print(f"Prediction for sample {sample}: {predicted:.3f}")
