import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import pennylane as qml

# Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define Quantum Device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def circuit(inputs):
    # Encode classical data into quantum states
    for i in range(len(inputs)):
        qml.RY(inputs[i], wires=i)
    
    # Example: Apply some gates (you can modify this part)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])

    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QuantumMLP(torch.nn.Module):
    def __init__(self):
        super(QuantumMLP, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)  # Input layer to hidden layer
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image
        x = torch.relu(self.fc1(x))
        
        # Pass through the quantum circuit - adjust inputs accordingly.
        outputs = []
        for i in range(len(x)):
            output = circuit(x[i].detach().numpy())
            outputs.append(output)
        
        return torch.tensor(outputs)

# Initialize Model and Optimizer
model = QuantumMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training Loop
for epoch in range(5):  # Number of epochs
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(images.float())
        
        loss = criterion(outputs.view(-1), labels) 
        loss.backward()
        
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

print("Training Complete")
