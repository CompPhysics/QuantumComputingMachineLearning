import numpy as np
from collections import Counter
from sklearn.datasets import fetch_openml
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")

# --- STEP 1: Load and preprocess MNIST zeros (4x4 binarized) ---

print("Downloading and preprocessing MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
X_zeros = X[y == '0'] / 255.0  # Normalize
X_zeros = X_zeros[:200]  # For speed

def downsample_binarize(img, size=4):
    img = img.reshape(28, 28)
    small = resize(img, (size, size), order=0, anti_aliasing=False, preserve_range=True)
    binary = (small > 0.5).astype(int)
    return ''.join(map(str, binary.flatten()))

samples_bin = [downsample_binarize(img) for img in X_zeros]
data_dist = Counter(samples_bin)
total = sum(data_dist.values())
data_dist = {k: v / total for k, v in data_dist.items()}

# --- STEP 2: Quantum Circuit Utils ---

# R_y rotation
def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2),  np.cos(theta/2)]
    ])

# CNOT gate for any 2 qubits
def CNOT(n, control, target):
    dim = 2**n
    op = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(np.binary_repr(i, width=n))
        if bits[control] == '1':
            bits[target] = '1' if bits[target] == '0' else '0'
        j = int(''.join(bits), 2)
        op[i, j] = 1
    return op

# Build the quantum state from params
def variational_state(params):
    n = len(params)
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1

    # Apply Ry rotations
    U = 1
    for theta in params:
        U = np.kron(U, Ry(theta))
    state = U @ state

    # Apply entangling CNOTs: linear chain
    for i in range(n - 1):
        state = CNOT(n, i, i + 1) @ state

    return state

# Sample bitstrings from state
def sample_state(psi, num_samples=1000):
    probs = np.abs(psi)**2
    states = [format(i, f'0{int(np.log2(len(psi)))}b') for i in range(len(psi))]
    return np.random.choice(states, size=num_samples, p=probs)

# Get distribution from samples
def get_prob_dist(samples):
    counts = Counter(samples)
    total = sum(counts.values())
    return {x: c / total for x, c in counts.items()}

# KL divergence: D_KL(p || q)
def kl_divergence(p, q, eps=1e-10):
    kl = 0.0
    for x in p:
        px = p[x]
        qx = q.get(x, eps)
        kl += px * np.log(px / (qx + eps))
    return kl

# Parameter-shift gradients
def parameter_shift_grad(params, data_dist, shift=np.pi/2, num_samples=500):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[i] += shift
        minus[i] -= shift

        psi_plus = variational_state(plus)
        psi_minus = variational_state(minus)
        dist_plus = get_prob_dist(sample_state(psi_plus, num_samples))
        dist_minus = get_prob_dist(sample_state(psi_minus, num_samples))

        kl_plus = kl_divergence(data_dist, dist_plus)
        kl_minus = kl_divergence(data_dist, dist_minus)
        grads[i] = 0.5 * (kl_plus - kl_minus)
    return grads

# --- STEP 3: Training VQBM on MNIST patterns ---

n_qubits = 4
params = np.random.uniform(0, 2*np.pi, size=n_qubits)
lr = 0.2

print("\nTraining VQBM...\n")
for step in range(100):
    psi = variational_state(params)
    model_samples = sample_state(psi, num_samples=1000)
    model_dist = get_prob_dist(model_samples)
    loss = kl_divergence(data_dist, model_dist)
    
    grads = parameter_shift_grad(params, data_dist)
    params -= lr * grads

    if step % 10 == 0:
        print(f"Step {step:3d}: KL Divergence = {loss:.5f}")

# --- STEP 4: Results ---

print("\nFinal learned distribution (top states):")
final_samples = sample_state(variational_state(params), num_samples=2000)
final_dist = get_prob_dist(final_samples)
for k, v in sorted(final_dist.items(), key=lambda x: -x[1])[:10]:
    print(f"{k}: {v:.4f}")
