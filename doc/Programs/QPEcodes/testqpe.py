import numpy as np

def quantum_phase_estimation(H, num_counting_qubits, initial_state=None):
    """
    Simulate the Quantum Phase Estimation algorithm to find eigenvalues of a Hermitian matrix H.
    
    Parameters:
        H (numpy.ndarray): Hermitian matrix (dimension N x N, typically N=2^n for n qubits in the target system).
        num_counting_qubits (int): Number of qubits in the counting register (controls precision of estimation).
        initial_state (numpy.ndarray or None): State vector for the target register (length N). 
            If None, a random state is chosen. For deterministic results, provide an eigenvector of H.
            
    Returns:
        list of (float, float): A list of tuples where each tuple is (estimated_eigenvalue, probability).
        If an eigenstate was provided as initial_state, one dominant eigenvalue with probability ~1 should appear.
    """
    # Ensure H is a numpy array and Hermitian
    H = np.array(H, dtype=complex)
    if not np.allclose(H, H.conj().T):
        raise ValueError("Input matrix H must be Hermitian.")
    N = H.shape[0]  # Dimension of the target register's state space
    
    # 1. Compute the unitary U = exp(2π i H).
    #    Because H is Hermitian, it has a spectral decomposition H = V diag(eigvals) V^†.
    #    We exponentiate the eigenvalues to get U's eigenvalues exp(2π i * eigval).
    eigvals, eigvecs = np.linalg.eigh(H)   # eigvals are real
    U = (eigvecs * np.exp(2j * np.pi * eigvals)) @ eigvecs.conj().T  # construct U = V * diag(exp(2πi λ)) * V^†
    # U is unitary and U|ψ_i> = exp(2π i λ_i)|ψ_i> for each eigenpair (λ_i, |ψ_i>) [oai_citation:12‡raw.githubusercontent.com](https://raw.githubusercontent.com/blueqat/blueqat-tutorials/refs/heads/master/tutorial/3_ftqc/02_pea2.ipynb#:~:text=%5C%5Clambda_j%5En%7D%7Bn%21%7D%5C%5Cleft%7C%5C%5Cpsi_j%5C%5Cright%5C%5Crangle%5C%5Cnonumber%5C%5C%5C%5C%5Cn,metadata).
    
    # 2. Prepare the initial state of the whole system (counting + target).
    M = 2 ** num_counting_qubits  # size of counting register state space
    # Counting register starts in |0...0>, target in given initial_state or random state.
    if initial_state is None:
        # If no initial state provided, choose a random state on the target (and normalize it).
        psi = np.random.rand(N) + 1j * np.random.rand(N)
        psi /= np.linalg.norm(psi)
    else:
        psi = np.array(initial_state, dtype=complex)
        psi /= np.linalg.norm(psi)  # normalize the initial state
    # Full initial state = |0...0>_counting ⊗ |psi>_target.
    full_state = np.zeros((M * N,), dtype=complex)
    # The counting register |0...0> corresponds to index 0, so place psi in that block of the state vector.
    full_state[0:N] = psi
    
    # 3. Apply Hadamard gates to all counting qubits.
    # A Hadamard on a counting qubit transforms |0> -> (|0>+|1>)/√2 and |1> -> (|0>-|1>)/√2.
    # We loop over each counting qubit to apply the tensor product of H with identity on others.
    for q in range(num_counting_qubits):
        # The qubit q has a bit-value that flips every 2^q indices in the counting register.
        # We iterate over the state vector and apply H to the amplitudes where qubit q is 0 and 1.
        step = 2 ** q          # half period for this qubit's flipping
        period = 2 ** (q + 1)  # full period for this qubit (0 then 1)
        half_block = step * N  # number of consecutive entries for which qubit q is fixed to 0 or 1, in the flat state vector
        new_state = full_state.copy()
        # Traverse the state in chunks where qubit q is constant
        for start in range(0, M * N, period * N):
            # In each period, first half_block entries have qubit q = 0, next half_block have qubit q = 1
            # Apply H: new_0 = (old_0 + old_1)/√2, new_1 = (old_0 - old_1)/√2
            end0 = start + half_block
            end1 = start + 2 * half_block
            psi0 = full_state[start:end0]       # segment where q is |0>
            psi1 = full_state[end0:end1]        # segment where q is |1>
            # Superpose the segments (note: psi0 and psi1 are of length half_block = step*N)
            new_state[start:end0] = (psi0 + psi1) / np.sqrt(2)
            new_state[end0:end1] = (psi0 - psi1) / np.sqrt(2)
        full_state = new_state
    # After this loop, the counting register is in an equal superposition of all 2^t states [oai_citation:13‡en.wikipedia.org](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm#:~:text=state%3AImage%3A%20%7B%5Cdisplaystyle%20%7C%5CPsi%20_%7B1%7D%5Crangle%20%3D%28H,1%7D%7Cj_%7B%5Cell).
    
    # 4. Apply controlled-U^{2^j} operations for each counting qubit j.
    # For each counting qubit, if that qubit is |1>, we apply U^(2^j) to the target register state.
    for q in range(num_counting_qubits):
        # Compute U^(2^q) efficiently using eigen-decomposition (square the phase).
        U_power = (eigvecs * np.exp(2j * np.pi * eigvals * (2 ** q))) @ eigvecs.conj().T
        # Now apply this controlled operation: loop over each basis state of counting register.
        new_state = full_state.copy()
        for idx in range(M):
            # Check if the q-th bit of idx is 1 (in binary representation)
            if (idx >> q) & 1:  # if the q-th bit of idx is 1
                # The portion of the state corresponding to counting index 'idx' is from idx*N to idx*N+N-1.
                start = idx * N
                end = start + N
                # Apply U_power to the target portion of the state vector for this counting state
                new_state[start:end] = U_power.dot(full_state[start:end])
        full_state = new_state
    # After this, the phase information is encoded in the amplitudes of the counting register states [oai_citation:14‡en.wikipedia.org](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm#:~:text=,2%5Cpi%20ik%5Ctheta%20%7D%7C%5Cpsi%20%5Crangle).
    
    # 5. Apply the inverse Quantum Fourier Transform (QFT) on the counting register.
    # We will construct the inverse QFT matrix on t qubits of size MxM and apply it to the counting part of full_state.
    M = 2 ** num_counting_qubits
    # Reshape state to (M, N) for convenience (each row = amplitude for a counting basis state, of length N for target).
    full_state = full_state.reshape(M, N)
    # Construct inverse QFT matrix F_dagger of size M x M (M = 2^t).
    F_dagger = np.zeros((M, M), dtype=complex)
    omega = np.exp(-2j * np.pi / M)  # primitive Mth root of unity (for inverse, use e^{-2πi/M})
    for x in range(M):
        for y in range(M):
            # Inverse QFT: |y> -> (1/√M) * Σ_x exp(-2π i * x * y / M) |x>
            F_dagger[x, y] = omega ** (x * y)
    F_dagger /= np.sqrt(M)
    # Apply the inverse QFT on the counting register (matrix multiply on the first index of full_state).
    full_state = F_dagger.dot(full_state)
    # Reshape back to flat state vector (optional).
    full_state = full_state.reshape(M * N)
    
    # 6. Measure the counting register (simulation by extracting probabilities).
    # Calculate the probability of each outcome k (0 <= k < 2^t) by summing probabilities of target states.
    probabilities = []
    for k in range(M):
        # Probability = sum of |amplitude|^2 of state where counting register = k
        start = k * N
        end = start + N
        prob = np.linalg.norm(full_state[start:end])**2
        if prob > 1e-12:  # consider only significant probabilities
            probabilities.append((k, prob))
    # Sort outcomes by probability descending
    probabilities.sort(key=lambda x: x[1], reverse=True)
    # Convert outcome index to eigenvalue estimate (fraction of 2^t)
    results = []
    for (k, prob) in probabilities:
        phase_est = k / (2 ** num_counting_qubits)  # in [0, 1)
        results.append((phase_est, prob))
    return results

# Example usage:
if __name__ == "__main__":
    # Define a 2x2 Hermitian matrix (Hamiltonian). For example:
    H = np.array([[0.25, 0.0],
                  [0.0, 0.5]])  # eigenvalues 0.25 and 0.5
    # Run QPE with 3 counting qubits (3 bits of precision).
    results = quantum_phase_estimation(H, num_counting_qubits=3)
    print("Estimated eigenvalues (as fractions of 1) and their probabilities:")
    for phase, prob in results:
        print(f"  Eigenvalue ≈ {phase:.3f} (fraction {phase} of 1) with probability {prob:.2f}")
