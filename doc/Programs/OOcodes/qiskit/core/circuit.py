from qiskit import QuantumCircuit, Aer, execute

class Circuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qc = QuantumCircuit(num_qubits)
    
    def add_gate(self, gate_func):
        gate_func(self.qc)
    
    def add_measure_all(self):
        self.qc.measure_all()
    
    def run_statevector(self):
        backend = Aer.get_backend('statevector_simulator')
        job = execute(self.qc, backend)
        result = job.result()
        statevector = result.get_statevector()
        return statevector
    
    def run_measurement(self, shots=1024):
        backend = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return counts
