import numpy as np
import qiskit as qk
import qiskit.opflow as opflow
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')



def hamiltonian_xxx():
    # define Heisenberg XXX hamiltonian opflow way
    # defining operators using qiskit opflow
    identity = opflow.I
    pauli_x = opflow.X
    pauli_y = opflow.Y
    pauli_z = opflow.Z

    x_interaction = (identity^pauli_x^pauli_x) + (pauli_x^pauli_x^identity)
    y_interaction = (identity^pauli_y^pauli_y) + (pauli_y^pauli_y^identity)
    z_interaction = (identity^pauli_z^pauli_z) + (pauli_z^pauli_z^identity)
    total_interaction = x_interaction + y_interaction + z_interaction

    return total_interaction



def propagator(time):
    # define the time evolution operator opflow way
    hamiltonian = hamiltonian_xxx()
    time_evolution_unitary = (time * hamiltonian).exp_i()

    return time_evolution_unitary



def classical_simulation(time, initial_state):
    # A copy paste from the notebook just to have it here

    probability = np.zeros_like(time_points)

    for i, t in enumerate(time_points):
        probability[i] = np.abs((~initial_state @ propagator(float(t)) \
                                    @ initial_state).eval())**2


    return probability




if __name__=='__main__':
    ket_zero = opflow.Zero
    ket_one = opflow.One
    one_one_zero = ket_one^ket_one^ket_zero
    one_zero_one = ket_one^ket_zero^ket_one
    zero_one_one = ket_zero^ket_one^ket_one
    end_time = np.pi
    time_points = np.linspace(0, end_time, 100)

    prob_110 = classical_simulation(time_points, one_one_zero)
    #prob_101 = classical_simulation(time_points, one_zero_one)
    #prob_011 = classical_simulation(time_points, zero_one_one)

    plt.plot(time_points, prob_110, label='110')
    #plt.plot(time_points, prob_101, label='101')
    #plt.plot(time_points, prob_011, label='011')

    plt.xlim(0, end_time)
    plt.ylim(0, 1)
    plt.xlabel('Time')
    plt.ylabel(r'Probability')
    plt.legend()
    #plt.title(r'Evolution of state $|110\rangle, |101\rangle, |011\rangle$ under $H_{XXX}$')
    plt.show()
