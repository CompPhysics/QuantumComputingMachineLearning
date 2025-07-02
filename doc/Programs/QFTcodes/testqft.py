import numpy as np

def qft_matrix(n):
    """Construct the Quantum Fourier Transform matrix for n qubits."""
    N = 2 ** n
    omega = np.exp(2j * np.pi / N)
    qft = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            qft[i, j] = omega ** (i * j)
    return qft / np.sqrt(N)

def apply_qft(state_vector):
    """Apply QFT to a given state vector using matrix multiplication."""
    N = len(state_vector)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("Length of state vector must be a power of 2.")
    QFT = qft_matrix(n)
    return QFT @ state_vector

def compare_qft_fft(n, basis_index=0):
    """Compare QFT and FFT on a computational basis state |basis_index>."""
    N = 2 ** n
    # Initialize the computational basis state |basis_index>
    state = np.zeros(N, dtype=complex)
    state[basis_index] = 1.0

    # Apply QFT
    qft_result = apply_qft(state)

    # Classical FFT (with normalization)
    fft_result = np.fft.fft(state) / np.sqrt(N)

    # Compare magnitudes and phases
    print(f"\nComparing QFT and FFT results for |{basis_index}> with {n} qubits:")
    for i in range(N):
        print(f"Index {i:2d}: QFT = {qft_result[i]:.4f}, FFT = {fft_result[i]:.4f}, "
              f"Difference = {abs(qft_result[i] - fft_result[i]):.2e}")

    return qft_result, fft_result

# Example usage
if __name__ == "__main__":
    n_qubits = 3  # try with 2, 3, 4, etc.
    for basis_idx in range(2 ** n_qubits):
        compare_qft_fft(n_qubits, basis_idx)
