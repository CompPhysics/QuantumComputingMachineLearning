"""Chapter 2, Secs. 2.7-2.8: the elementary quantum gates as matrices.

Single-qubit gates are 2x2 unitary matrices, two-qubit gates 4x4, all in the
computational basis |0>, |1> (and |00>, |01>, |10>, |11> with the first
qubit as the most significant bit).  Rotation gates follow the convention
R_sigma(theta) = exp(-i theta sigma / 2), so that they rotate the Bloch
vector by the angle theta about the corresponding axis.

Shared by every chapter; imported as ``from gates import X, H, CNOT, ...``.
"""
import numpy as np

# ------------------------------------------------------------ single qubit
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)                 # phase gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # pi/8 gate

PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


def Rx(theta):
    """Rotation about the x axis of the Bloch sphere, exp(-i theta X/2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def Ry(theta):
    """Rotation about the y axis, exp(-i theta Y/2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta):
    """Rotation about the z axis, exp(-i theta Z/2)."""
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])


def Rn(theta, n):
    """Rotation by theta about the unit vector n = (n_x, n_y, n_z)."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    sigma_n = n[0] * X + n[1] * Y + n[2] * Z
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * sigma_n


def phase(lam):
    """The phase gate P(lambda) = diag(1, e^{i lambda}); S = P(pi/2), T = P(pi/4)."""
    return np.array([[1, 0], [0, np.exp(1j * lam)]])


def U3(theta, phi, lam):
    """The general single-qubit gate U(theta, phi, lambda), Eq. (2.U3)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


# ------------------------------------------------------------- two qubits
def controlled(U):
    """Controlled-U on two qubits: |0><0| (x) I + |1><1| (x) U.

    The first (most significant) qubit is the control."""
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    return np.kron(P0, I) + np.kron(P1, U)


CNOT = controlled(X)
CZ = controlled(Z)
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)
TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[6, 6], TOFFOLI[7, 7], TOFFOLI[6, 7], TOFFOLI[7, 6] = 0, 0, 1, 1


# --------------------------------------------------------------- helpers
def is_unitary(U, tol=1e-12):
    U = np.asarray(U)
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=tol)


def is_hermitian(A, tol=1e-12):
    A = np.asarray(A)
    return np.allclose(A, A.conj().T, atol=tol)


def commutator(A, B):
    return A @ B - B @ A


def anticommutator(A, B):
    return A @ B + B @ A


def kron(*ops):
    """Tensor product of several operators (or states), left to right."""
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def equal_up_to_phase(U, V, tol=1e-10):
    """True if U = e^{i alpha} V for some real alpha."""
    U, V = np.asarray(U), np.asarray(V)
    idx = np.unravel_index(np.argmax(np.abs(V)), V.shape)
    if abs(V[idx]) < tol:
        return np.allclose(U, V, atol=tol)
    alpha = U[idx] / V[idx]
    return abs(abs(alpha) - 1) < tol and np.allclose(U, alpha * V, atol=tol)


if __name__ == "__main__":
    # Properties quoted in Sec. 2.2.4 and Sec. 2.7
    for name, P in PAULIS.items():
        assert is_unitary(P) and is_hermitian(P) and np.allclose(P @ P, I)
    assert np.allclose(commutator(X, Y), 2j * Z)
    assert np.allclose(commutator(Y, Z), 2j * X)
    assert np.allclose(commutator(Z, X), 2j * Y)
    assert np.allclose(anticommutator(X, Y), 0)
    assert np.allclose(H @ Z @ H, X) and np.allclose(H @ X @ H, Z)
    assert np.allclose(S @ S, Z) and np.allclose(T @ T, S)
    assert equal_up_to_phase(Rz(np.pi), Z) and equal_up_to_phase(Rx(np.pi), X)
    # ZYZ decomposition, Eq. (2.zyz)
    th, ph, la = 0.7, 1.9, -0.4
    assert equal_up_to_phase(Rz(ph) @ Ry(th) @ Rz(la), U3(th, ph, la))
    # CNOT truth table and SWAP from three CNOTs
    for x in range(2):
        for y in range(2):
            v = np.zeros(4); v[2 * x + y] = 1
            w = CNOT @ v
            assert w[2 * x + (y ^ x)] == 1
    cnot_rev = kron(H, H) @ CNOT @ kron(H, H)      # CNOT with control and target exchanged
    assert np.allclose(CNOT @ cnot_rev @ CNOT, SWAP)
    print("all gate identities verified")
    print("determinants of X, Y, Z:", [round(float(np.linalg.det(P).real), 6) for P in (X, Y, Z)])
