# Quantum Kernel SVM on Iris (Setosa vs Versicolor) using PennyLane and scikit-learn

import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset and select two classes (0: setosa, 1: versicolor)
iris = datasets.load_iris()
X = iris.data
y = iris.target
mask = y != 2  # drop class '2' (virginica)
X = X[mask]
y = y[mask]

# Standardize features and reduce to 2 dimensions via PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
   X_reduced, y, test_size=0.2, random_state=42
)
# Define quantum device and feature map (angle encoding on 2 qubits)
n_qubits = 2
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def feature_map(x):
   # Encode 2 features into rotation angles
   for i in range(n_qubits):
       qml.RY(x[i] * np.pi, wires=i)
   # Optional: add entanglement (e.g., ZZ interaction)
   qml.CNOT(wires=[0, 1])
   qml.RZ((x[0] + x[1]) * np.pi, wires=1)
   qml.CNOT(wires=[0, 1])
   return qml.state()

# Compute quantum kernel (fidelity) between two feature vectors
def quantum_kernel(x1, x2):
   # Compute state vectors for each input
   state1 = feature_map(x1)
   state2 = feature_map(x2)
   # Kernel = |<phi(x1)|phi(x2)>|^2
   overlap = np.vdot(state1, state2)
   return np.abs(overlap) ** 2

# Build kernel (Gram) matrices for training and test sets
n_train = len(X_train)
n_test = len(X_test)
kernel_train = np.zeros((n_train, n_train))
for i in range(n_train):
   for j in range(n_train):
       kernel_train[i, j] = quantum_kernel(X_train[i], X_train[j])

kernel_test = np.zeros((n_test, n_train))
for i in range(n_test):
   for j in range(n_train):
       kernel_test[i, j] = quantum_kernel(X_test[i], X_train[j])

# Train SVM with precomputed quantum kernel
svm = SVC(kernel='precomputed')
svm.fit(kernel_train, y_train)

# Predict on test set and evaluate
y_pred = svm.predict(kernel_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
