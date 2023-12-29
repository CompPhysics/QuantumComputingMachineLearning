"""
Implementing exact quantum state tomography for one qubit

Author: Linus Ekstrøm 08.09.22
"""
import numpy as np
import matplotlib.pyplot as plt
import qiskit.visualization as vis
from qiskit.visualization.bloch import Bloch



def inner_product(a, b):
    """
    computes <a|b>

    Inputs:
        a: numpy matrix representing a column vector
        b: numpy matrix representing a column vector

    Returns:
        numpy matrix representing the inner product
    """
    return np.dot(a.conj().T, b)



def outer_product(a, b):
    """
    computes |a><b|

    Inputs:
        a: numpy matrix representing a column vector
        b: numpy matrix representing a column vector

    Returns:
        numpy matrix representing the outer product
    """
    return np.outer(a, b.conj().T)



def bloch_vector_from_density_matrix(rho):
    """
    calculates bloch coordinates a_x, a_y, a_z from
    rho = 1/2 * (I + a_x*sigma_x + a_y*sigma_y + a_z*sigma_z)
    """
    a_x = rho[0][1] + rho[1][0]                                                 # feel like there might be factor of 1/2 missing here
    a_y = 1j * (rho[0][1] - rho[1][0])
    a_z = rho[0][0] - rho[1][1]

    bloch_vector = np.array(
        [[a_x], [a_y], [a_z]],
        dtype=complex,
    )
    return bloch_vector



def density_matrix_from_bloch_vector(bloch_vector):
    """
    inverse of the above function
    """
    rho = np.zeros((2, 2), dtype=complex)
    rho[0][0] = 1 + bloch_vector[2]
    rho[1][1] = 1 - bloch_vector[2]
    rho[0][1] = bloch_vector[0] - 1j*bloch_vector[1]
    rho[1][0] = bloch_vector[0] + 1j*bloch_vector[1]

    return 0.5 * rho


def plot_bloch_vectors(bloch_vectors):
    """
    wrapper function for easy plotting
    """
    bloch_sphere = Bloch(figsize=(15, 12))
    bloch_sphere.ylpos = [1.1, -1.2]
    bloch_sphere.xlabel = [
        '$\\left|0\\right>+\\left|1\\right>$',
        '$\\left|0\\right>-\\left|1\\right>$'
    ]
    bloch_sphere.ylabel = [
        '$\\left|0\\right>+i\\left|1\\right>$',
        '$\\left|0\\right>-i\\left|1\\right>$'
    ]
    bloch_vectors = [vector.real for vector in bloch_vectors]
    bloch_sphere.add_vectors(bloch_vectors)
    bloch_sphere.render()
    bloch_sphere.fig
    plt.show()



def find_orthogonal_compliment(a):
    """
    use the Householder transformation to find orthogonal compliment of vector

    need to look into if I can just do this directly on the basis vectors
    directly
    """
    identity = np.eye(len(a))
    householder_matrix = identity - 2 * outer_product(a, a)
    orthogonal_compliment = np.matmul(householder_matrix, a)

    return orthogonal_compliment



def row_reduce_matrix(B, tol=1e-8, debug=False):
    """
    code from https://gist.github.com/sgsfak/77a1c08ac8a9b0af77393b24e44c9547
    updated print statements to python 3, might rewrite for readability later
    """

    A = B.copy()
    rows, cols = A.shape
    r = 0
    pivots_pos = []
    row_exchanges = np.arange(rows)
    for c in range(cols):
        if debug:
            print(f"Now at row {r}, and col {c}, with matrix: {A}")
        ## Find the pivot row:
        pivot = np.argmax (np.abs (A[r:rows,c])) + r
        m = np.abs(A[pivot, c])
        if debug:
            print(f"Found pivot {m} in row {pivot}")
        if m <= tol:
            ## Skip column c, making sure the approximately zero terms are
            ## actually zero.
            A[r:rows, c] = np.zeros(rows-r)
            if debug:
                print(f"All elements at and below ({r}, {c}) are zero.. moving on..")
        else:
            ## keep track of bound variables
            pivots_pos.append((r,c))

            if pivot != r:
                ## Swap current row and pivot row
                A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
                row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]

            if debug:
                print(f"Swap row {r} with row {pivot} Now: {A}")

            ## Normalize pivot row
            A[r, c:cols] = A[r, c:cols] / A[r, c];

            ## Eliminate the current column
            v = A[r, c:cols]
            ## Above (before row r):
            if r > 0:
                ridx_above = np.arange(r)
                A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
                if debug:
                    print(f"Elimination above performed: {A}")
            ## Below (after row r):
            if r < rows-1:
                ridx_below = np.arange(r+1,rows)
                A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
                if debug:
                    print(f"Elimination below performed: {A}")
            r += 1
            ## Check if done
        if r == rows:
            break;
    return (A, pivots_pos, row_exchanges)



def check_axes_linear_independence(measurement_axes, tolerance=1e-8):
    """
    checks if the measurement axes represented by matrices axis_1, axis_2,
    axis_3 in list measurement_axes are linearly independent
    """
    axis_1, axis_2, axis_3 = measurement_axes
    """
    #print((np.round(axis_1), np.round(axis_2), np.round(axis_3))                # not so sure about this one
    matrix = np.ones((4,4), dtype=complex)
    matrix[:, 0] = axis_1.reshape(4,).T
    matrix[:, 1] = axis_2.reshape(4,).T
    matrix[:, 2] = axis_3.reshape(4,).T
    print(np.linalg.det(matrix))
    """
    # We can make the same matrix and row reduce i think
    matrix = np.zeros((4,4), dtype=complex)
    matrix[:, 0] = np.eye(2).reshape(4,).T
    matrix[:, 1] = axis_1.reshape(4,).T
    matrix[:, 2] = axis_2.reshape(4,).T
    matrix[:, 3] = axis_3.reshape(4,).T

    determinant = np.linalg.det(matrix)
    print(determinant)
    #row_reduced_matrix, _, _ = row_reduce_matrix(matrix)
    #print(row_reduced_matrix)
    #row_reduced_matrix = row_reduced_matrix[:, :-1]
    #determinant = np.linalg.det(row_reduced_matrix)

    return np.abs(determinant) > tolerance




def generate_measurement_axes(a, b, c):
    """
    generates tau_i operators representing the measurement axes defined by
    vectors a, b, c and their orthogonal compliments

    Inputs:
        a:  numpy matrix representing a column vector
        b:  numpy matrix representing a column vector
        c:  numpy matrix representing a column vector

    Returns:
        numpy matrices representing tau_i operators in up, down basis
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_c = np.linalg.norm(c)
    if not norm_a <= 1 or not norm_b <= 1 or not norm_c <= 1:                   # this looks stupid
        print('Norms must be unit or less')
        print(f'norm_a: {norm_a}, norm_b: {norm_b}, norm_c: {norm_c}')
        return None

    outer_a = outer_product(a, a)
    outer_b = outer_product(b, b)
    outer_c = outer_product(c, c)

    #print(outer_a)
    #print(outer_b)
    #print(outer_c)
    #print(bloch_vector_from_density_matrix(outer_a))
    #print(bloch_vector_from_density_matrix(outer_b))
    #print(bloch_vector_from_density_matrix(outer_c))

    bloch_a = bloch_vector_from_density_matrix(outer_a)                         # need to do the convertion if I want to plot anyway
    bloch_b = bloch_vector_from_density_matrix(outer_b)
    bloch_c = bloch_vector_from_density_matrix(outer_c)
    bloch_list = [bloch_a, bloch_b, bloch_c]
    #bloch_list = [bloch_a]
    #print(bloch_list)
    #plot_bloch_vectors(bloch_list)
    orthogonal_list = []
    for vector in bloch_list:
        compliment = find_orthogonal_compliment(vector)
        orthogonal_list.append(compliment)

    plot_list = bloch_list + orthogonal_list
    plot_bloch_vectors(plot_list)

    #bra_one_ket_one = density_matrix_from_bloch_vector(orthogonal_list[0])
    #print(bra_one_ket_one)

    measurement_operators = []                                                  # this way of doing it is obviously not optimized
    for i in range(len(bloch_list)):
        outer_psi_i = density_matrix_from_bloch_vector(bloch_list[i])
        #print(psi_i)
        #outer_psi_i = outer_product(psi_i, psi_i)
        outer_psi_i_perp = density_matrix_from_bloch_vector(orthogonal_list[i])
        #outer_psi_i_perp = outer_product(psi_i_perp, psi_i_perp)
        #print(outer_psi_i_perp)
        tau_i = outer_psi_i - outer_psi_i_perp
        measurement_operators.append(tau_i)

    assert check_axes_linear_independence(measurement_operators), \
        f'chosen axes not linearly independent, the method will not work.'

    measurement_operators.insert(0, np.eye(2))

    return measurement_operators



def convert_T_parameters_to_S():
    """
    function to perform the matrix inversion problem of obtaining stokes
    parameters S_i from the Stokes-like parameters T_i


    leaving this for later.
    """


    return



def calculate_stokes_parameters_from_measurement(
    axis_1_outcomes,
    axis_2_outcomes,
    axis_3_outcomes,
    ):
    """
    uses measurement probabilities to calculate obtain stokes parameters
    """
    stokes_parameters = np.zeros((4,))
    stokes_parameters[0] = 1
    stokes_parameters[1] = axis_1_outcomes[0] - axis_1_outcomes[1]
    stokes_parameters[2] = axis_2_outcomes[0] - axis_2_outcomes[1]
    stokes_parameters[3] = axis_3_outcomes[0] - axis_3_outcomes[1]

    return stokes_parameters



def construct_density_matrix(measurement_axes, stokes_parameters):
    density_matrix = np.zeros((2, 2), dtype=complex)
    for i, axis in enumerate(measurement_axes):
        density_matrix += stokes_parameters[i] * axis

    return 0.5 * density_matrix





if __name__=='__main__':
    ket_zero = np.array(
        [[1], [0]],
        dtype=complex,
    )
    ket_one = np.array(
        [[0], [1]],
        dtype=complex,
    )
    psi_1 = ket_zero
    #factor = 1/5
    #psi_2 = (np.sqrt(factor)*ket_zero + np.sqrt(1 - factor)*ket_one)
    psi_2 = 1 / np.sqrt(2) * (ket_zero + ket_one)
    psi_3 = 1 / np.sqrt(2) * (ket_zero + 1j*ket_one)

    #print(inner_product(ket_zero, ket_one))
    #print((outer_product(ket_zero, ket_one))
    rho_1 = outer_product(psi_1, psi_1)
    rho_2 = outer_product(psi_2, psi_2)
    rho_3 = outer_product(psi_3, psi_3)
    bloch_1 = bloch_vector_from_density_matrix(rho_1)
    bloch_2 = bloch_vector_from_density_matrix(rho_2)
    bloch_3 = bloch_vector_from_density_matrix(rho_3)

    #bloch_list = [bloch_1, bloch_2, bloch_3]
    #plot_bloch_vectors(bloch_list)


    measurement_axes = generate_measurement_axes(psi_1, psi_2, psi_3)
    #for tau_i in measurement_axes:
    #    print(tau_i)

    ############################### TEST CASE 1 ###############################
    # Measurements corresponding to |psi> = 1/sqrt(2)(|0>+|1>)
    axis_1_outcomes = (1.0, 0.0)                                                #measuring in z direction
    axis_2_outcomes = (1.0, 0.0)
    axis_3_outcomes = (1.0, 0.0)
    stokes_parameters = calculate_stokes_parameters_from_measurement(
        axis_1_outcomes,
        axis_2_outcomes,
        axis_3_outcomes,
    )
    density_matrix = construct_density_matrix(
        measurement_axes,
        stokes_parameters,
    )
    print(density_matrix)

    ############################### TEST CASE 1 ###############################
    # Measurements corresponding to |psi> = 1/sqrt(2)(|0>+i|1>)
    axis_1_outcomes = (0.5, 0.5)                                                #measuring in z direction
    axis_2_outcomes = (0.5, 0.5)
    axis_3_outcomes = (1.0, 0.0)
    stokes_parameters = calculate_stokes_parameters_from_measurement(
        axis_1_outcomes,
        axis_2_outcomes,
        axis_3_outcomes,
    )
    density_matrix = construct_density_matrix(
        measurement_axes,
        stokes_parameters,
    )
    print(density_matrix)
