from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and preprocess dataset
dataset = load_iris()
X = dataset.data
y = dataset.target

# For simplicity, we'll only classify between two classes
X = X[y != 2]
y = y[y != 2]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define quantum feature map
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)

# Set up quantum instance
quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=1024)

# Initialize QSVC
qsvc = QSVC(quantum_instance=quantum_instance, feature_map=feature_map)

# Train QSVC
qsvc.fit(X_train, y_train)

# Evaluate QSVC
accuracy = qsvc.score(X_test, y_test)
print(f'QSVC accuracy: {accuracy * 100:.2f}%')
