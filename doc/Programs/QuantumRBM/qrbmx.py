import numpy as np
from collections import Counter
from itertools import product

# Pauli matrices
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# R_y gate
def Ry(theta):
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ])

# CNOT gate for arbitrary control and target
def CNOT(n, control, target):
    dim = 2 ** n
    op = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(np.binary_repr(i, width=n))
        if bits[control] == '1':
            bits[target] = '1' if bits[target] == '0' else '0'
        j = int("".join(bits), 2)
        op[i, j] = 1
    return op

# Create n-qubit variational state
def variational_state(params):
    n = len(params)
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1  # |00...0⟩

    # Apply R_y on each qubit
    U = 1
    for theta in params:
        U = np.kron(U, Ry(theta))
    state = U @ state

    # Apply entangling CNOTs (pairwise)
    for i in range(n - 1):
        state = CNOT(n, i, i + 1) @ state

    return state

# Sample from state
def sample_state(psi, num_samples=1000):
    probs = np.abs(psi) ** 2
    states = [format(i, f"0{int(np.log2(len(psi)))}b") for i in range(len(psi))]
    return np.random.choice(states, size=num_samples, p=probs)

# KL divergence: D_KL(p || q) = sum_x p(x) log(p(x)/q(x))
def kl_divergence(p_data, p_model, eps=1e-10):
    kl = 0.0
    for x in p_data:
        p = p_data[x]
        q = p_model.get(x, eps)
        kl += p * np.log(p / (q + eps))
    return kl

# Get empirical probabilities from samples
def get_prob_dist(samples):
    counts = Counter(samples)
    total = sum(counts.values())
    return {x: c / total for x, c in counts.items()}


# Parameter shift gradient
def parameter_shift_grad(params, data_dist, shift=np.pi/2, num_samples=500):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[i] += shift
        minus[i] -= shift

        psi_plus = variational_state(plus)
        psi_minus = variational_state(minus)

        model_plus = get_prob_dist(sample_state(psi_plus, num_samples))
        model_minus = get_prob_dist(sample_state(psi_minus, num_samples))

        kl_plus = kl_divergence(data_dist, model_plus)
        kl_minus = kl_divergence(data_dist, model_minus)

        grads[i] = 0.5 * (kl_plus - kl_minus)
    return grads


# === Synthetic Dataset ===
data_samples = ['000', '001', '011', '111'] * 50  # Synthetic repeated patterns
data_dist = get_prob_dist(data_samples)

# === VQBM Training ===
n_qubits = 3
params = np.random.uniform(0, 2*np.pi, size=n_qubits)
lr = 0.1

for step in range(1000):
    psi = variational_state(params)
    model_samples = sample_state(psi, num_samples=500)
    model_dist = get_prob_dist(model_samples)
    loss = kl_divergence(data_dist, model_dist)

    grads = parameter_shift_grad(params, data_dist)
    params -= lr * grads

    if step % 10 == 0:
        print(f"Step {step:3d}: KL Divergence = {loss:.6f}")

# Final Results
print("\nFinal learned distribution:")
psi = variational_state(params)
samples = sample_state(psi, num_samples=1000)
final_model_dist = get_prob_dist(samples)
for k in sorted(final_model_dist):
    print(f"{k}: {final_model_dist[k]:.3f}")
