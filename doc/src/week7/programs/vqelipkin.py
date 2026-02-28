#!/usr/bin/env python3
"""
FINAL CORRECTED VQE for Lipkin Model
All bugs fixed:
1. ✓ Proper CNOT gates
2. ✓ 4-layer ansatz  
3. ✓ Correct Y measurement: H·S† (NOT H·S, NOT S†·H)
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import minimize

np.random.seed(42)

print("="*70)
print("LIPKIN MODEL VQE - FINAL CORRECTED VERSION")
print("="*70)

# ============================================================================
# Setup
# ============================================================================

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron(*args):
    result = args[0]
    for mat in args[1:]:
        result = np.kron(result, mat)
    return result

def pauli_3qubit(label):
    pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    return kron(pauli_dict[label[0]], pauli_dict[label[1]], pauli_dict[label[2]])

# Gates
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)  # S†

def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Rz(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

CNOT_2qubit = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# ============================================================================
# Hamiltonian
# ============================================================================

def construct_5D_operators(j):
    dim = 5
    m_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
    Jz_5D = np.diag(m_vals).astype(complex)
    Jp_5D = np.zeros((dim, dim), dtype=complex)
    for i, m in enumerate(m_vals[:-1]):
        Jp_5D[i+1, i] = np.sqrt(j*(j+1) - m*(m+1))
    Jm_5D = Jp_5D.T.conj()
    return Jz_5D, Jp_5D, Jm_5D, m_vals

def embed_5D_to_8D(operator_5D):
    operator_8D = np.zeros((8, 8), dtype=complex)
    operator_8D[:5, :5] = operator_5D
    return operator_8D

def construct_lipkin_8D(j, epsilon, V, W):
    N = 2 * j
    Jz_5D, Jp_5D, Jm_5D, _ = construct_5D_operators(j)
    H_5D = np.zeros((5, 5), dtype=complex)
    H_5D += (epsilon / 2.0) * (Jz_5D - (N/2) * np.eye(5, dtype=complex))
    H_5D += (V / 2.0) * (Jp_5D @ Jp_5D + Jm_5D @ Jm_5D)
    H_5D += (W / 2.0) * (Jz_5D @ Jz_5D)
    H_8D = embed_5D_to_8D(H_5D)
    return H_8D

j = 2.0
H_lipkin_8D = construct_lipkin_8D(j, epsilon=1.0, V=0.5, W=0.0)
eigenvalues, eigenvectors = np.linalg.eigh(H_lipkin_8D)
E_exact = eigenvalues[0]
psi_exact = eigenvectors[:, 0]

print(f"\nExact ground state energy: {E_exact:.10f}")

# Pauli decomposition
def decompose_to_pauli_basis(operator_8D, threshold=1e-10):
    pauli_labels = ['I', 'X', 'Y', 'Z']
    pauli_decomp = []
    for p0, p1, p2 in product(pauli_labels, repeat=3):
        label = p0 + p1 + p2
        pauli_op = pauli_3qubit(label)
        coeff = np.trace(operator_8D @ pauli_op) / 8.0
        if abs(coeff) > threshold:
            pauli_decomp.append((coeff, label))
    return pauli_decomp

H_pauli = decompose_to_pauli_basis(H_lipkin_8D)
print(f"Pauli terms: {len(H_pauli)}")

# ============================================================================
# CORRECTED Measurement
# ============================================================================

def measure_pauli_term(psi, pauli_label, n_shots):
    """
    FINAL CORRECT VERSION
    Key: Use H·S† for Y measurement (H·Sdg where Sdg = S†)
    """
    rotation = np.eye(8, dtype=complex)
    
    for qubit_idx, pauli_char in enumerate(pauli_label):
        if pauli_char == 'X':
            gate = H_gate
        elif pauli_char == 'Y':
            gate = H_gate @ Sdg  # CRITICAL: H·S† is correct!
        else:
            continue
        
        if qubit_idx == 0:
            rotation = kron(gate, I, I) @ rotation
        elif qubit_idx == 1:
            rotation = kron(I, gate, I) @ rotation
        else:
            rotation = kron(I, I, gate) @ rotation
    
    psi_rotated = rotation @ psi
    probs = np.abs(psi_rotated)**2
    probs = probs / np.sum(probs)
    
    measurements = np.random.choice(8, size=n_shots, p=probs)
    
    expectation = 0.0
    for outcome in range(8):
        count = np.sum(measurements == outcome)
        if count == 0:
            continue
        
        binary = f"{outcome:03b}"
        eigenvalue = 1.0
        for qubit_idx, (pauli_char, bit) in enumerate(zip(pauli_label, binary)):
            if pauli_char != 'I' and bit == '1':
                eigenvalue *= -1
        
        expectation += eigenvalue * count / n_shots
    
    return expectation

def measure_energy(psi, H_pauli, n_shots):
    energy = 0.0
    for coeff, label in H_pauli:
        if label == 'III':
            energy += coeff.real
        else:
            energy += coeff.real * measure_pauli_term(psi, label, n_shots)
    return energy

# Test measurement
E_measured = measure_energy(psi_exact, H_pauli, 100000)
print(f"Measurement test: {E_measured:.6f} (should be {E_exact:.6f})")

# ============================================================================
# Ansatz
# ============================================================================

def ansatz_4layers(params):
    U = np.eye(8, dtype=complex)
    param_idx = 0
    
    for layer in range(4):
        U = kron(Ry(params[param_idx]), I, I) @ U
        param_idx += 1
        U = kron(I, Ry(params[param_idx]), I) @ U
        param_idx += 1
        U = kron(I, I, Ry(params[param_idx])) @ U
        param_idx += 1
        
        U = kron(CNOT_2qubit, I) @ U
        U = kron(I, CNOT_2qubit) @ U
        
        U = kron(Rz(params[param_idx]), I, I) @ U
        param_idx += 1
        U = kron(I, Rz(params[param_idx]), I) @ U
        param_idx += 1
        U = kron(I, I, Rz(params[param_idx])) @ U
        param_idx += 1
    
    return U

psi_0 = np.zeros(8, dtype=complex)
psi_0[0] = 1.0

# ============================================================================
# VQE
# ============================================================================

print(f"\nRunning VQE (exact evaluation)...")

def cost(params):
    psi = ansatz_4layers(params) @ psi_0
    pauli_op = pauli_3qubit('III')
    E = np.vdot(psi, H_lipkin_8D @ psi).real
    return E

result = minimize(cost, np.random.uniform(-np.pi, np.pi, 24),
                 method='BFGS', options={'maxiter': 200, 'disp': False})

print(f"VQE result (exact): {result.fun:.10f}")
print(f"Error: {abs(result.fun - E_exact):.6e}")

# With shots
print(f"\nRunning VQE with shots...")
def cost_shots(params):
    psi = ansatz_4layers(params) @ psi_0
    return measure_energy(psi, H_pauli, 10000)

result_shots = minimize(cost_shots, np.random.uniform(-np.pi, np.pi, 24),
                       method='BFGS', options={'maxiter': 100, 'disp': False})

print(f"VQE result (10k shots): {result_shots.fun:.10f}")
print(f"Error: {abs(result_shots.fun - E_exact):.6e}")

if result_shots.fun >= E_exact - 0.01:
    print(f"\n✓ VARIATIONAL PRINCIPLE SATISFIED")
else:
    print(f"\n✗ Below ground state (bug still present)")

print("\n" + "="*70)
