import numpy as np
import matplotlib.pyplot as plt

# Qiskit and Qiskit Nature imports
from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.circuit.library import TwoLocal
from qiskit_nature.drivers import UnitsType, PySCFDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.circuit.library import HartreeFock

# Define the range of H-H distances (in Angstrom)
bond_distances = np.arange(0.5, 2.6, 0.1)  # 0.5 Å to 2.5 Å in 0.1 Å increments
energies = []  # to store ground state energy for each distance

# Loop over distances and compute ground-state energy via VQE
for dist in bond_distances:
    # 1. Define H2 molecule at the given bond distance
    molecule = f"H 0 0 0; H 0 0 {dist}"  # two H atoms on z-axis separated by 'dist'
    driver = PySCFDriver(atom=molecule, unit=UnitsType.ANGSTROM, basis="sto3g", charge=0, spin=0)
    problem = ElectronicStructureProblem(driver)

    # 2. Build second-quantized Hamiltonian
    second_q_ops = problem.second_q_ops()
    main_op = second_q_ops["ElectronicEnergy"]  # electronic Hamiltonian (no nuclear repulsion added yet)

   # 3. Prepare qubit mapping (Parity) with two-qubit reduction for spin symmetry
    particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
    num_particles = (particle_number.num_alpha, particle_number.num_beta)
    num_spin_orbitals = particle_number.num_spin_orbitals
    mapper = ParityMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)
    qubit_op = converter.convert(main_op, num_particles=num_particles)

    # 4. Setup the VQE algorithm (ansatz, initial state, optimizer, backend)
    # Hartree-Fock initial state
    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
    # TwoLocal ansatz with RY and RZ rotations and CZ entanglement, with HF initial state
    ansatz = TwoLocal(qubit_op.num_qubits, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')
    ansatz = ansatz.compose(init_state, front=True)  # prepend HF state


    optimizer = L_BFGS_B()  # optimizer for classical parameter updates
    backend = Aer.get_backend('aer_simulator_statevector')  # statevector simulator backend
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=backend)

    # 5. Run VQE to obtain the minimum eigenvalue (ground state energy)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    # Add the nuclear repulsion energy to get total molecular energy
    electronic_result = problem.interpret(result)
    energy_hartree = electronic_result.total_energies[0]  # ground state energy in Hartree
    energies.append(energy_hartree)
    print(f"Distance = {dist:.2f} Å, Ground state energy = {energy_hartree:.6f} Hartree")

# 6. Plot the ground state energy as a function of bond length
plt.figure(figsize=(6,4))
plt.plot(bond_distances, energies, marker='o', color='orange', label='Ground State Energy')
plt.xlabel('H-H Bond Length (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('H$_2$ Ground State Energy vs Bond Length (STO-3G)')
plt.legend()
plt.grid(True)
plt.show()
