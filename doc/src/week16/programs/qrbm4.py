import pennylane as qml
import numpy as np
# Number of visible and hidden qubits
n_v, n_h = 2, 1
dev = qml.device('default.qubit', wires=n_v+n_h)
# Define a variational circuit (QNode)
@qml.qnode(dev, interface='autograd')
def circuit(params):
# params is a vector of rotation angles
# Prepare all qubits in |0>
# Example ansatz: one layer of rotations + entangling gates
    for i in range(n_v + n_h):
        qml.RY(params[i], wires=i)
# entangle visible to hidden
        for i in range(n_v):
            qml.CNOT(wires=[i, n_v])  # connect each visible i to hidden n_v
    return qml.probs(wires=list(range(n_v)))

# Example target distribution over 2 visible bits
target = np.array([0.3, 0.2, 0.1, 0.4])  # must sum to 1
def loss(params):
    probs = circuit(params)  # model probabilities for visible states
# Add small epsilon to avoid log(0)
    return np.sum(target * np.log((target + 1e-9) / probs))
def parameter_shift_grad(params):
    grads = np.zeros_like(params)
    shift = np.pi/2
    for idx in range(len(params)):
        shift_vector = np.zeros_like(params)
        shift_vector[idx] = shift
        probs_plus = circuit(params + shift_vector)
        probs_minus = circuit(params - shift_vector)
        loss_plus  = np.sum(target * np.log((target + 1e-9) / probs_plus))
        loss_minus = np.sum(target * np.log((target + 1e-9) / probs_minus))
        grads[idx] = 0.5 * (loss_plus - loss_minus)
    return grads

#Initialize parameters and perform a simple gradient descent
params = np.random.normal(0, 0.1, size=(n_v+n_h,))
learning_rate = 0.1
for epoch in range(100):
    grads = parameter_shift_grad(params)
    params -= learning_rate * grads
