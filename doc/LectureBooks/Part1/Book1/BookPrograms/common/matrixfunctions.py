"""Matrix exponentials and logarithms without SciPy.

expm_hermitian(H)      exp(-i t H) style via the spectral theorem, Eq. (2.fofA)
expm_antihermitian(A)  exp(A) for A^dagger = -A, via the Hermitian matrix iA
expm(A)                any square matrix, scaling and squaring with a Taylor series

Shared by chapters 3-8; ``from matrixfunctions import expm, expm_hermitian``.
"""
import numpy as np


def expm_hermitian(H, scale=1.0):
    """exp(scale * H) for Hermitian H (scale may be complex, e.g. -i t)."""
    w, V = np.linalg.eigh(H)
    return (V * np.exp(scale * w)) @ V.conj().T


def expm_antihermitian(A):
    """exp(A) for anti-Hermitian A: A = -i (iA) with iA Hermitian."""
    return expm_hermitian(1j * A, scale=-1j)


def expm(A, terms=30):
    """exp(A) for a general square matrix by scaling and squaring."""
    A = np.asarray(A, dtype=complex)
    norm = np.linalg.norm(A, 1)
    s = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0 else 0
    B = A / 2 ** s
    result = np.eye(A.shape[0], dtype=complex)
    term = np.eye(A.shape[0], dtype=complex)
    for k in range(1, terms):
        term = term @ B / k
        result = result + term
        if np.abs(term).max() < 1e-18:
            break
    for _ in range(s):
        result = result @ result
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    H = M + M.conj().T
    assert np.allclose(expm_hermitian(H, -1j * 0.7), expm(-1j * 0.7 * H))
    A = M - M.conj().T
    U = expm_antihermitian(A)
    assert np.allclose(U @ U.conj().T, np.eye(5)) and np.allclose(U, expm(A))
    print("matrix-function checks passed")
