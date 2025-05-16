import numpy as np
from collections import Counter
from sklearn.datasets import fetch_openml
from skimage.transform import resize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- STEP 1: Load and preprocess MNIST zeros (4x4 binarized) ---

print("Downloading and preprocessing MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
X_zeros = X[y == '0'] / 255.0
X_zeros = X_zeros[:200]

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

def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2),  np.cos(theta/2)]
    ])

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

def variational_state(params):
    n = len(params)
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    U = 1
    for theta in params:
        U = np.kron(U, Ry(theta))
    state = U @ state
    for i in range(n - 1):
        state = CNOT(n, i, i + 1) @ state
    return state

def sample_state(psi, num_samples=1000):
    probs = np.abs(psi)**2
    states = [format(i, f'0{int(np.log2(len(psi)))}b') for i in range(len(psi))]
    return np.random.choice(states, size=num_samples, p=probs)

def get_prob_dist(samples):
    counts = Counter(samples)
    total = sum(counts.values())
    return {x: c / total for x, c in counts.items()}

# --- Contrastive Divergence Loss ---

def energy(bitstring, psi):
    index = int(bitstring, 2)
    prob = np.abs(psi[index])**2
    return -np.log(prob + 1e-10)

def contrastive_divergence_loss(psi, data_samples, model_samples):
    E_data = np.mean([energy(x, psi) for x in data_samples])
    E_model = np.mean([energy(x, psi) for x in model_samples])
    return E_data - E_model

def parameter_shift_grad_cd(params, data_samples, shift=np.pi/2, num_samples=500):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[i] += shift
        minus[i] -= shift

        psi_plus = variational_state(plus)
        psi_minus = variational_state(minus)

        model_plus = sample_state(psi_plus, num_samples)
        model_minus = sample_state(psi_minus, num_samples)

        loss_plus = contrastive_divergence_loss(psi_plus, data_samples, model_plus)
        loss_minus = contrastive_divergence_loss(psi_minus, data_samples, model_minus)

        grads[i] = 0.5 * (loss_plus - loss_minus)
    return grads

# --- STEP 3: Training ---

n_qubits = 4
params = np.random.uniform(0, 2*np.pi, size=n_qubits)
lr = 0.1
data_samples = samples_bin[:100]

print("\nTraining VQBM with Contrastive Divergence...\n")
for step in range(100):
    psi = variational_state(params)
    model_samples = sample_state(psi, num_samples=500)
    loss = contrastive_divergence_loss(psi, data_samples, model_samples)

    grads = parameter_shift_grad_cd(params, data_samples)
    params -= lr * grads

    if step % 10 == 0:
        print(f"Step {step:3d}: CD Loss = {loss:.4f}")

# --- STEP 4: Plot Results ---

psi_final = variational_state(params)
samples = sample_state(psi_final, 2000)
model_dist = get_prob_dist(samples)

# Top k states
top_k = 10
all_states = list(set(list(data_dist.keys()) + list(model_dist.keys())))
top_states = sorted(all_states, key=lambda s: data_dist.get(s, 0) + model_dist.get(s, 0), reverse=True)[:top_k]

x = np.arange(len(top_states))
data_vals = [data_dist.get(s, 0) for s in top_states]
model_vals = [model_dist.get(s, 0) for s in top_states]

plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, data_vals, width=0.4, label="Data")
plt.bar(x + 0.2, model_vals, width=0.4, label="Model")
plt.xticks(x, top_states, rotation=45)
plt.ylabel("Probability")
plt.title("Top Learned Distributions: Data vs VQBM Model")
plt.legend()
plt.tight_layout()
plt.show()
