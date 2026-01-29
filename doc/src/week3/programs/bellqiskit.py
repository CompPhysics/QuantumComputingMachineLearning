from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def bell_circuit(label: str) -> QuantumCircuit:
    """
    Create a circuit that prepares a Bell state from |00> and measures both qubits.
    label in {"phi+", "phi-", "psi+", "psi-"}.
    """
    qc = QuantumCircuit(2, 2)

    # Core: create Phi+
    qc.h(0)
    qc.cx(0, 1)

    # Local adjustments
    if label == "phi+":
        pass
    elif label == "phi-":
        qc.z(0)          # add relative phase
    elif label == "psi+":
        qc.x(1)          # swap |00>+|11> -> |01>+|10>
    elif label == "psi-":
        qc.x(1)
        qc.z(0)
    else:
        raise ValueError("Unknown label. Use: phi+, phi-, psi+, psi-")

    # Measure both qubits
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def run_on_ibm(shots: int = 4096):
    # Uses saved IBM Quantum account (e.g. via `qiskit-ibm-runtime` setup)
    service = QiskitRuntimeService()

    # Pick a real device (set simulator=True if you want a simulator)
    backend = service.least_busy(operational=True, simulator=False)

    circuits = [
        bell_circuit("phi+"),
        bell_circuit("phi-"),
        bell_circuit("psi+"),
        bell_circuit("psi-"),
    ]

    # Transpile to backend ISA
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = [pm.run(qc) for qc in circuits]

    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=shots)
    results = job.result()

    labels = ["phi+", "phi-", "psi+", "psi-"]
    for lab, res in zip(labels, results):
        # counts stored in classical register "c" by default
        counts = res.data.c.get_counts()
        print(f"\nBackend: {backend.name} | Bell state: {lab}")
        print("Counts:", counts)


if __name__ == "__main__":
    run_on_ibm(shots=4096)
