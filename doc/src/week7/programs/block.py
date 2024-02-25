import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Generate a random quantum state vector
num_qubits = 1  # Number of qubits
state = rand_ket(2 ** num_qubits)

# Alternatively, visualize the state using a Bloch sphere representation
bloch = Bloch()
bloch.add_states(state)
bloch.show()

# Show the plots
plt.show()
