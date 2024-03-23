from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.x(0)
qc.draw('mpl')

from qiskit.visualization import visualize_transition
visualize_transition(qc)

qc = QuantumCircuit(1)
qc.x(0)
qc.y(0)
qc.h(0)
qc.draw('mpl')
visualize_transition(qc)
