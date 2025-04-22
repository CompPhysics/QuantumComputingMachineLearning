import pennylane as qml
from pennylane import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Create a quantum device with 2 qubits
dev = qml.device("default.qubit", wires=2)

# Define quantum feature map (encoding classical data into quantum states)
def feature_map(x):
   qml.Hadamard(wires=0)
   qml.Hadamard(wires=1)
   qml.RZ(x[0], wires=0)
   qml.RZ(x[1], wires=1)
   qml.CNOT(wires=[0, 1])
   qml.RY(x[0], wires=0)
   qml.RY(x[1], wires=1)

# Variational ansatz (simple circuit to be trained)
def variational_circuit(params):
   qml.RY(params[0], wires=0)
   qml.RY(params[1], wires=1)
   qml.CNOT(wires=[0, 1])
   qml.RZ(params[2], wires=1)

# Quantum node
@qml.qnode(dev, interface="torch")
def circuit(x, weights):
   feature_map(x)
   variational_circuit(weights)
   return qml.expval(qml.PauliZ(0))

# Create a torch-compatible quantum layer
class QuantumLayer(nn.Module):
   def __init__(self):
       super().__init__()
       # Initialize trainable parameters
       self.weights = nn.Parameter(0.01 * torch.randn(3))

   def forward(self, x):
       # Apply quantum circuit to each input in the batch
       return torch.stack([circuit(x[i], self.weights) for i in range(x.shape[0])])

# Define the full model
class QSVM(nn.Module):
   def __init__(self):
       super().__init__()
       self.q_layer = QuantumLayer()
       self.classifier = nn.Linear(1, 1)

   def forward(self, x):
       q_out = self.q_layer(x).unsqueeze(1)  # Add dimension for linear layer
       return torch.sigmoid(self.classifier(q_out))

# Example toy dataset (linearly separable)
X = torch.tensor([[0.1, 0.2], [1.2, 0.9], [0.2, 0.1], [1.0, 1.1]], dtype=torch.float32)
Y = torch.tensor([[0.], [1.], [0.], [1.]], dtype=torch.float32)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Instantiate model, loss, and optimizer
model = QSVM()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
   for xb, yb in loader:
       pred = model(xb)
       loss = criterion(pred, yb)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
