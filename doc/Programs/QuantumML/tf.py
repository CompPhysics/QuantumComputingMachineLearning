import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset and normalize it
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data (resizing and normalizing)
X_train = X_train.reshape(-1, 28, 28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28, 28).astype(np.float32) / 255.0

# Use binary encoding for the labels (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Quantum Circuit Layer
def create_quantum_model(input_shape):
   qubits = cirq.GridQubit.rect(1, 4)  # Create a 1x4 grid of qubits
   circuit = cirq.Circuit()

   # Quantum feature map (simple)
   for qubit in qubits:
       circuit.append(cirq.rx(np.pi/2)(qubit))  # Rotate each qubit

   # Add entangling gates
   circuit.append(cirq.CNOT(qubits[0], qubits[1]))
   circuit.append(cirq.CNOT(qubits[2], qubits[3]))

   return circuit, qubits

# TensorFlow Quantum Model
def build_quantum_model(input_shape):
   # Quantum circuit for encoding input data
   input_data = layers.Input(shape=(28, 28), dtype=tf.float32)

   # Convert classical image input into quantum data using TFQ
   quantum_circuit, qubits = create_quantum_model(input_shape)

   # Apply quantum circuit to input data (simulation)
   quantum_layer = tfq.layers.AddCircuit()(input_data)

   # Output layer
   output = layers.Dense(10, activation='softmax')(quantum_layer)

   # Define and compile the model
   model = Model(inputs=input_data, outputs=output)
   model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

   return model

# Build and train the quantum model
quantum_model = build_quantum_model((28, 28))
quantum_model.summary()

# Training the quantum model
quantum_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the quantum model
test_loss, test_acc = quantum_model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


