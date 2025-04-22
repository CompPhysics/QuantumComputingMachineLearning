import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)

# Quantum device with 2 qubits
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum circuit (variational ansatz)
def quantum_circuit(inputs, weights):
   # Encode classical data
   for i in range(n_qubits):
       qml.RY(inputs[i], wires=i)

   # Trainable layer
   qml.CNOT(wires=[0, 1])
   for i in range(n_qubits):
       qml.Rot(*weights[i], wires=i)

# QNode: quantum node that can be called like a function
@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
   quantum_circuit(inputs, weights)
   return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Torch module for quantum layer
class QuantumLayer(nn.Module):
   def __init__(self):
       super().__init__()
       # Initialize trainable parameters for each qubit
       self.q_weights = nn.Parameter(0.01 * torch.randn(n_qubits, 3))

   def forward(self, x):
       # Apply quantum circuit to each input sample
       return torch.stack([qnode(x[i], self.q_weights) for i in range(x.shape[0])])

# Full hybrid quantum-classical neural network
class HybridQNN(nn.Module):
   def __init__(self):
       super().__init__()
       self.quantum_layer = QuantumLayer()
       self.classifier = nn.Sequential(
           nn.Linear(n_qubits, 4),
           nn.ReLU(),
           nn.Linear(4, 1),
           nn.Sigmoid()
       )

   def forward(self, x):
       q_out = self.quantum_layer(x)
       return self.classifier(q_out)

# Toy dataset (binary classification)
X = torch.tensor([[0.0, 0.1], [0.1, 0.2], [3.0, 3.1], [3.1, 3.0]], dtype=torch.float32)
Y = torch.tensor([[0.], [0.], [1.], [1.]], dtype=torch.float32)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Instantiate model, loss, optimizer
model = HybridQNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(30):
   for xb, yb in loader:
       pred = model(xb)
       loss = criterion(pred, yb)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
