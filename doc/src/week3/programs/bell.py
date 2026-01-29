import math
from typing import List

ComplexVec = List[complex]
ComplexMat = List[List[complex]]

# -------- Linear algebra helpers (small matrices only) --------

def matmul(A: ComplexMat, B: ComplexMat) -> ComplexMat:
    n, m, p = len(A), len(B[0]), len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)] for i in range(n)]

def matvec(A: ComplexMat, v: ComplexVec) -> ComplexVec:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def kron(A: ComplexMat, B: ComplexMat) -> ComplexMat:
    # Kronecker product
    out = []
    for i in range(len(A)):
        for ii in range(len(B)):
            row = []
            for j in range(len(A[0])):
                for jj in range(len(B[0])):
                    row.append(A[i][j] * B[ii][jj])
            out.append(row)
    return out

def normalize(v: ComplexVec) -> ComplexVec:
    norm = math.sqrt(sum((z.conjugate() * z).real for z in v))
    if norm == 0:
        return v
    return [z / norm for z in v]

def pretty_state(v: ComplexVec, tol: float = 1e-12) -> str:
    # Basis ordering: |00>,|01>,|10>,|11>
    labels = ["|00>", "|01>", "|10>", "|11>"]
    terms = []
    for amp, lab in zip(v, labels):
        if abs(amp) > tol:
            # format complex nicely
            a = amp.real
            b = amp.imag
            if abs(b) < tol:
                terms.append(f"{a:+.6f}{lab}")
            elif abs(a) < tol:
                terms.append(f"{b:+.6f}i{lab}")
            else:
                terms.append(f"({a:+.6f}{b:+.6f}i){lab}")
    return " ".join(terms) if terms else "0"

# -------- Single-qubit gates --------

inv_sqrt2 = 1.0 / math.sqrt(2.0)

I2: ComplexMat = [[1+0j, 0+0j],
                  [0+0j, 1+0j]]

X: ComplexMat = [[0+0j, 1+0j],
                 [1+0j, 0+0j]]

Z: ComplexMat = [[1+0j,  0+0j],
                 [0+0j, -1+0j]]

H: ComplexMat = [[inv_sqrt2,  inv_sqrt2],
                 [inv_sqrt2, -inv_sqrt2]]

# -------- Two-qubit gates (4x4) in basis |00>,|01>,|10>,|11> --------

# CNOT with control qubit 0 (MSB) and target qubit 1 (LSB)
CNOT_01: ComplexMat = [
    [1+0j, 0+0j, 0+0j, 0+0j],
    [0+0j, 1+0j, 0+0j, 0+0j],
    [0+0j, 0+0j, 0+0j, 1+0j],
    [0+0j, 0+0j, 1+0j, 0+0j],
]

def apply_single_qubit(gate2: ComplexMat, which: int) -> ComplexMat:
    # which=0 applies to qubit 0, which=1 applies to qubit 1
    if which == 0:
        return kron(gate2, I2)
    elif which == 1:
        return kron(I2, gate2)
    else:
        raise ValueError("which must be 0 or 1")

def make_bell_states() -> None:
    # Start from |00> = [1,0,0,0]^T
    ket00: ComplexVec = [1+0j, 0+0j, 0+0j, 0+0j]

    # Core circuit for Phi+ : (H on qubit0) then CNOT(0->1)
    U_phi_plus = matmul(CNOT_01, apply_single_qubit(H, 0))
    phi_plus = normalize(matvec(U_phi_plus, ket00))

    # Other Bell states by local gates
    # Phi- = (Z on qubit 0) Phi+
    phi_minus = normalize(matvec(apply_single_qubit(Z, 0), phi_plus))
    # Psi+ = (X on qubit 1) Phi+
    psi_plus = normalize(matvec(apply_single_qubit(X, 1), phi_plus))
    # Psi- = (Z on qubit 0)(X on qubit 1) Phi+  (order doesn't matter here since local)
    psi_minus = normalize(matvec(matmul(apply_single_qubit(Z, 0), apply_single_qubit(X, 1)), phi_plus))

    print("Basis ordering: |00>, |01>, |10>, |11>\n")
    print("Phi+  =", pretty_state(phi_plus))
    print("Phi-  =", pretty_state(phi_minus))
    print("Psi+  =", pretty_state(psi_plus))
    print("Psi-  =", pretty_state(psi_minus))

if __name__ == "__main__":
    make_bell_states()
