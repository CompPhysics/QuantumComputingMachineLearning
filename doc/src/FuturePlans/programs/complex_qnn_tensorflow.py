
import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
y = y.reshape(-1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Quantum circuit parameters
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Define the QNN layer using a variational circuit
def qnn_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.CNOT(wires=[0, 1])
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (n_qubits,)}

qlayer = qml.qnn.KerasLayer(qml.QNode(qnn_circuit, dev, interface="tf", diff_method="parameter-shift"),
                            weight_shapes, output_dim=1)

# Build a hybrid quantum-classical model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    qlayer,
    tf.keras.layers.Activation("sigmoid")
])

#tf.keras.optimizers.legacy.Adam

# Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.1),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
