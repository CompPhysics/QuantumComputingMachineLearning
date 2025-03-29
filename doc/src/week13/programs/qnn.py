import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the quantum circuit layer
class QuantumCircuitLayer(nn.Module):
    def __init__(self, n_qubits):
        """
        Initialize the quantum layer with trainable parameters for rotations.
        """
        super(QuantumCircuitLayer, self).__init__()
        self.n_qubits = n_qubits
        # Each qubit has 3 rotation parameters: Rx, Ry, Rz
        self.theta = nn.Parameter(torch.randn(n_qubits, 3))  # Shape (n_qubits, 3)

    def forward(self, x):
        """
        Forward pass to simulate a quantum circuit.
        """
        # Normalize input to [0, 2π] for quantum circuit
        x = (x % (2 * np.pi))
        
        # Simulate quantum state with parameterized rotations
        # (In reality, this is a simulation)
        quantum_state = torch.zeros(self.n_qubits)
        
        for i in range(self.n_qubits):
            # Apply rotations: Rx, Ry, Rz
            rotation = torch.sin(self.theta[i, 0] * x + self.theta[i, 1]) + torch.cos(self.theta[i, 2])
            quantum_state[i] = rotation  # Output is a real value for this simulation
        
        return quantum_state

# Define the Quantum Neural Network (QNN)
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, n_hidden, n_output):
        """
        Define a QNN with a quantum layer and classical layers.
        """
        super(QuantumNeuralNetwork, self).__init__()
        self.quantum_layer = QuantumCircuitLayer(n_qubits)
        
        # Classical fully connected layers
        self.fc1 = nn.Linear(n_qubits, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        # Pass input through quantum layer
        q_output = self.quantum_layer(x)
        
        # Pass quantum output through classical layers
        x = self.fc1(q_output)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create synthetic binary classification dataset
def generate_data(n_samples=1000):
    """
    Generate simple data for binary classification.
    """
    X = torch.rand((n_samples,)) * 2 * np.pi  # Inputs in [0, 2π]
    y = torch.where(torch.sin(X) > 0, 1.0, 0.0)  # Labels based on sine function
    return X.unsqueeze(1), y.unsqueeze(1)

# Set hyperparameters
n_qubits = 4
n_hidden = 8
n_output = 1
learning_rate = 0.01
epochs = 100
batch_size = 32

# Initialize the QNN model
model = QuantumNeuralNetwork(n_qubits, n_hidden, n_output)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate dataset
X_train, y_train = generate_data()

# Training loop
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_train):.4f}")

# Testing with a sample input
test_input = torch.tensor([[np.pi / 2]])  # Input π/2
test_output = model(test_input)
print(f"\nTest Input: {test_input.item()} - QNN Output: {test_output.item():.4f}")
