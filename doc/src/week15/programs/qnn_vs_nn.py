#!/usr/bin/env python3
"""
Classical Neural Network vs Quantum Neural Network
===================================================
Pure PyTorch — no PennyLane, Qiskit, or any quantum library.

Task
----
Regression on the noisy polynomial

    f(x) = x³ - 2x² + 0.5x + 1  +  ε,    ε ~ N(0, 0.15²),  x ∈ [-2, 2]

Both models are trained with the same data split, loss (MSE), and optimiser
(Adam) so the comparison is fair.

Classical NN
------------
Standard MLP:  1 → 32 → 32 → 16 → 1
Activation: Tanh throughout (smooth, bounded — natural for bounded regression).

Quantum Neural Network (QNN)
-----------------------------
A variational quantum circuit simulated entirely in PyTorch matrix algebra.
No quantum libraries are used: gates are explicit 2×2 (and 4×4) complex
matrices, state evolution is matrix-vector multiplication, and the computation
graph carries gradients through all rotation angles via torch.autograd.

Architecture: data re-uploading (Pérez-Salinas et al. 2020).
  • n_qubits = 2, n_layers = 4
  • Each layer:
      1. Encode x → R_y(x * w_enc) on every qubit   (trainable scale w_enc)
      2. Trainable R_x(θ), R_y(φ), R_z(λ) on every qubit
      3. CNOT entanglement layer (fixed, qubit 0 → qubit 1)
  • Output: ⟨Z₀⟩ = expectation value of Pauli-Z on qubit 0, scaled by a
    trainable scalar and shifted by a trainable bias.

Re-uploading makes the input appear at every layer, which is known to give
single- and two-qubit circuits the expressibility of a Fourier series, making
them universal approximators for 1-D functions.

Contents
--------
  §1   Data generation
  §2   Classical neural network
  §3   Quantum gate library (pure PyTorch)
  §4   Quantum neural network
  §5   Shared training loop
  §6   Training both models
  §7   Evaluation and comparison
  §8   Visualisation (4-panel figure)
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device         : {device}")
print(f"PyTorch version: {torch.__version__}")

SEP = "=" * 68

# =============================================================================
# §1 — DATA GENERATION
# =============================================================================

def true_function(x: np.ndarray) -> np.ndarray:
    """Target polynomial  f(x) = x³ - 2x² + 0.5x + 1."""
    return x**3 - 2.0*x**2 + 0.5*x + 1.0


N_TRAIN  = 150
N_VAL    = 50
N_TEST   = 100
NOISE_SD = 0.15
X_LOW, X_HIGH = -2.0, 2.0

rng = np.random.default_rng(42)

x_all = rng.uniform(X_LOW, X_HIGH, N_TRAIN + N_VAL + N_TEST)
y_all = true_function(x_all) + rng.normal(0, NOISE_SD, len(x_all))

x_tr = x_all[:N_TRAIN];   y_tr = y_all[:N_TRAIN]
x_va = x_all[N_TRAIN:N_TRAIN+N_VAL];   y_va = y_all[N_TRAIN:N_TRAIN+N_VAL]
x_te = x_all[N_TRAIN+N_VAL:];          y_te = y_all[N_TRAIN+N_VAL:]

# Normalise input to [-1, 1] — important for both NN and QNN
x_mean = (X_LOW + X_HIGH) / 2
x_std  = (X_HIGH - X_LOW) / 2

def normalise(x): return (x - x_mean) / x_std

def to_tensor(x, y):
    xt = torch.tensor(normalise(x), dtype=torch.float32).unsqueeze(1).to(device)
    yt = torch.tensor(y,            dtype=torch.float32).unsqueeze(1).to(device)
    return xt, yt

X_tr, Y_tr = to_tensor(x_tr, y_tr)
X_va, Y_va = to_tensor(x_va, y_va)
X_te, Y_te = to_tensor(x_te, y_te)

# Dense grid for plotting the learned function
x_plot   = np.linspace(X_LOW, X_HIGH, 400)
y_exact  = true_function(x_plot)
X_plot_t = torch.tensor(normalise(x_plot), dtype=torch.float32).unsqueeze(1).to(device)

print(f"\nDataset:  {N_TRAIN} train | {N_VAL} val | {N_TEST} test")
print(f"Target:   f(x) = x³ - 2x² + 0.5x + 1,  noise σ = {NOISE_SD}")
print(f"Input normalised to [-1, 1]")

# =============================================================================
# §2 — CLASSICAL NEURAL NETWORK
# =============================================================================

class ClassicalNN(nn.Module):
    """
    Standard MLP regressor:  1 → 32 → 32 → 16 → 1.

    Architecture choices:
    - Tanh activations: smooth and bounded, well-suited to a bounded
      regression target; avoids the dying-neuron issue of ReLU on small data.
    - Width 32/32/16: enough capacity for a degree-3 polynomial without
      over-parameterisation on 150 training points.
    - No BatchNorm: unnecessary overhead on such a small dataset.

    Parameters: 1×32 + 32 + 32×32 + 32 + 32×16 + 16 + 16×1 + 1 = 1649
    """
    def __init__(self, hidden: tuple = (32, 32, 16)):
        super().__init__()
        layers = []
        d_in = 1
        for h in hidden:
            layers.append(nn.Linear(d_in, h))
            layers.append(nn.Tanh())
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (batch, 1)


classical_model = ClassicalNN().to(device)
n_classical = sum(p.numel() for p in classical_model.parameters()
                  if p.requires_grad)
print(f"\nClassical NN parameters: {n_classical:,}")
print(classical_model)

# =============================================================================
# §3 — QUANTUM GATE LIBRARY  (pure PyTorch, no quantum libraries)
# =============================================================================
#
# Every quantum gate is a matrix multiply on the full state vector.
# For n qubits the state lives in C^(2^n).
#
# Single-qubit rotation gates:
#   R_x(θ) = [[cos(θ/2),  -i sin(θ/2)],
#              [-i sin(θ/2),  cos(θ/2)]]
#   R_y(φ) = [[cos(φ/2), -sin(φ/2)],
#              [sin(φ/2),  cos(φ/2)]]
#   R_z(λ) = [[exp(-iλ/2), 0         ],
#              [0,           exp(iλ/2)]]
#
# Two-qubit CNOT (control=0, target=1):
#   [[1,0,0,0],
#    [0,1,0,0],
#    [0,0,0,1],
#    [0,0,1,0]]
#
# Applying a single-qubit gate G to qubit k of an n-qubit system:
#   Full matrix = I ⊗...⊗ G ⊗...⊗ I   (G at position k)
# implemented via Kronecker products.

def rx(theta: torch.Tensor) -> torch.Tensor:
    """R_x(θ): rotation around X-axis by angle θ.  Returns (2,2) complex tensor."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    return torch.stack([
        torch.stack([c,              -1j * s], dim=-1),
        torch.stack([-1j * s,        c      ], dim=-1),
    ], dim=-2)   # (..., 2, 2)


def ry(phi: torch.Tensor) -> torch.Tensor:
    """R_y(φ): rotation around Y-axis by angle φ.  Returns (2,2) complex tensor."""
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    return torch.stack([
        torch.stack([c,   -s], dim=-1),
        torch.stack([s,    c], dim=-1),
    ], dim=-2)


def rz(lam: torch.Tensor) -> torch.Tensor:
    """R_z(λ): rotation around Z-axis by angle λ.  Returns (2,2) complex tensor."""
    e_neg = torch.exp(-0.5j * lam)
    e_pos = torch.exp( 0.5j * lam)
    zero  = torch.zeros_like(lam)
    return torch.stack([
        torch.stack([e_neg, zero ], dim=-1),
        torch.stack([zero,  e_pos], dim=-1),
    ], dim=-2)


def kron2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Kronecker (tensor) product of two 2-D matrices A (m×n) and B (p×q).
    Returns (mp × nq) matrix.
    """
    m, n = A.shape
    p, q = B.shape
    return (A.unsqueeze(1).unsqueeze(3)
             * B.unsqueeze(0).unsqueeze(2)).reshape(m*p, n*q)


def single_qubit_gate_full(gate: torch.Tensor,
                            qubit: int,
                            n_qubits: int) -> torch.Tensor:
    """
    Embed a (2,2) single-qubit gate into the full 2^n_qubits space.

    Constructs  I ⊗ ... ⊗ gate ⊗ ... ⊗ I
    where gate acts on `qubit` (0 = most significant).
    """
    I2   = torch.eye(2, dtype=gate.dtype, device=gate.device)
    mats = [gate if k == qubit else I2 for k in range(n_qubits)]
    out  = mats[0]
    for m in mats[1:]:
        out = kron2(out, m)
    return out   # (2^n, 2^n)


def cnot_gate(n_qubits: int,
              control: int,
              target: int,
              device_: torch.device) -> torch.Tensor:
    """
    CNOT gate in the full 2^n_qubits space.

    Built by explicit projection: act on |ctrl=0> subspace as identity
    and on |ctrl=1> subspace as X on the target qubit.
    """
    dim = 2 ** n_qubits
    U   = torch.zeros(dim, dim, dtype=torch.complex64, device=device_)
    for state in range(dim):
        bits  = [(state >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        if bits[control] == 0:
            U[state, state] = 1.0          # identity when control = 0
        else:
            # Flip target bit
            new_bits = bits.copy()
            new_bits[target] = 1 - new_bits[target]
            new_state = sum(b << (n_qubits - 1 - k)
                            for k, b in enumerate(new_bits))
            U[new_state, state] = 1.0      # X on target when control = 1
    return U

# =============================================================================
# §4 — QUANTUM NEURAL NETWORK  (data re-uploading architecture)
# =============================================================================
#
# Architecture (Pérez-Salinas et al. 2020, "Data re-uploading for a
# universal quantum classifier", Quantum 4, 226):
#
#   |0⟩^⊗n
#    │
#    ├── Layer 1:  Encode(x)  →  Rotate(θ₁)  →  Entangle
#    ├── Layer 2:  Encode(x)  →  Rotate(θ₂)  →  Entangle
#    │   ...
#    └── Layer L:  Encode(x)  →  Rotate(θ_L)  →  Entangle
#    │
#    └── Measure ⟨Z₀⟩  →  scale * ⟨Z₀⟩ + bias
#
# Encode(x):   R_y(w_enc_k * x) on each qubit k
#              (w_enc_k are trainable encoding weights)
# Rotate(θ):   R_x(θ_k) R_y(φ_k) R_z(λ_k) on each qubit k
# Entangle:    CNOT(0 → 1), CNOT(1 → 0)  (ring topology for n=2)
#
# The output is the expectation value ⟨ψ|Z⊗I|ψ⟩, where Z = diag(1,-1).
# A trainable scale and bias map this from [-1,1] to the target range.
#
# Why data re-uploading?
# A quantum circuit with fixed input encoding is a linear function of the
# state amplitudes after the final layer — thus a polynomial of fixed degree.
# Re-uploading the input at every layer increases the effective Fourier degree
# and makes the circuit a universal approximator.

class QuantumLayer(nn.Module):
    """
    One layer of the variational quantum circuit:
        Encode(x) → Rotate → Entangle

    Parameters (all real, trainable):
      enc_weights : (n_qubits,)   scaling of x before R_y encoding
      rx_weights  : (n_qubits,)   R_x rotation angles
      ry_weights  : (n_qubits,)   R_y rotation angles
      rz_weights  : (n_qubits,)   R_z rotation angles
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits

        # Encoding weights — initialised near 1 so encoding ≈ R_y(x)
        self.enc_weights = nn.Parameter(torch.ones(n_qubits) +
                                        0.1 * torch.randn(n_qubits))
        # Variational angles — initialised near 0
        self.rx_weights  = nn.Parameter(0.1 * torch.randn(n_qubits))
        self.ry_weights  = nn.Parameter(0.1 * torch.randn(n_qubits))
        self.rz_weights  = nn.Parameter(0.1 * torch.randn(n_qubits))

    def forward(self,
                psi: torch.Tensor,
                x:   torch.Tensor) -> torch.Tensor:
        """
        Apply one circuit layer to the state batch.

        Parameters
        ----------
        psi : (batch, 2^n_qubits)  complex state vector
        x   : (batch, 1)           scalar input (normalised)

        Returns
        -------
        psi_out : (batch, 2^n_qubits)  updated state
        """
        B = psi.shape[0]
        n = self.n_qubits

        # ── Encoding gate: R_y(w_enc_k * x) on each qubit ────────────────
        # enc_angles[b, k] = w_enc_k * x[b]
        enc_angles = x * self.enc_weights.unsqueeze(0)  # (B, n)
        for k in range(n):
            G    = ry(enc_angles[:, k])   # (B, 2, 2)
            Gfull = torch.stack([
                single_qubit_gate_full(G[b].to(psi.dtype), k, n)
                for b in range(B)
            ])  # (B, 2^n, 2^n)
            psi = torch.bmm(Gfull, psi.unsqueeze(-1)).squeeze(-1)

        # ── Variational gates: R_x R_y R_z on each qubit ─────────────────
        for k in range(n):
            for angle, gate_fn in [
                (self.rx_weights[k], rx),
                (self.ry_weights[k], ry),
                (self.rz_weights[k], rz),
            ]:
                G     = gate_fn(angle.unsqueeze(0).expand(B))  # (B, 2, 2)
                Gfull = torch.stack([
                    single_qubit_gate_full(G[b].to(psi.dtype), k, n)
                    for b in range(B)
                ])
                psi = torch.bmm(Gfull, psi.unsqueeze(-1)).squeeze(-1)

        # ── Entanglement: CNOT ring ───────────────────────────────────────
        # Precompute CNOT (constant, no gradient needed)
        if not hasattr(self, '_cnot_cache') or self._cnot_cache.device != psi.device:
            # ring: 0→1, then 1→0 for n=2
            cnot_01 = cnot_gate(n, 0, 1, psi.device)
            cnot_10 = cnot_gate(n, 1, 0, psi.device)
            self._cnot_cache = cnot_10 @ cnot_01   # combined (2^n, 2^n)

        C   = self._cnot_cache.to(psi.dtype)
        psi = (C @ psi.unsqueeze(-1)).squeeze(-1)  # broadcast over batch

        return psi


class QNN(nn.Module):
    """
    Variational Quantum Neural Network (data re-uploading).

    Architecture: n_qubits=2, n_layers=4.
    Input:  scalar x ∈ [-1, 1]
    Output: scale * ⟨Z₀⟩ + bias   (scalar regression output)

    State initialisation: |0...0⟩ = [1, 0, 0, ..., 0]ᵀ

    Measurement:
      ⟨Z₀⟩ = ⟨ψ|Z⊗I|ψ⟩ = Σ_{s: bit0(s)=0} |ψ_s|² - Σ_{s: bit0(s)=1} |ψ_s|²
      where bit0(s) is the most-significant bit of the basis state index s.

    Parameters per layer: n_qubits * 4 (enc + rx + ry + rz)
    Total trainable: n_layers * n_qubits * 4 + 2 (scale, bias)

    Notes on differentiability
    --------------------------
    All rotation angles are torch.nn.Parameter objects.  The gate matrices
    are constructed from these parameters via torch.cos/sin/exp, so
    autograd traces through the entire circuit.  The CNOT gate is a constant
    matrix (no parameters), so it only contributes a linear map in the graph.
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim      = 2 ** n_qubits

        self.layers = nn.ModuleList([
            QuantumLayer(n_qubits) for _ in range(n_layers)
        ])

        # Output scaling: map ⟨Z₀⟩ ∈ [-1,1] to the target range
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias  = nn.Parameter(torch.tensor(0.0))

        # Pauli-Z expectation operator on qubit 0:
        # Z₀ = Z ⊗ I = diag(+1, +1, -1, -1) for 2 qubits
        Z0 = torch.ones(self.dim)
        for s in range(self.dim):
            msb = (s >> (n_qubits - 1)) & 1
            if msb == 1:
                Z0[s] = -1.0
        self.register_buffer('Z0', Z0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1)  normalised scalar input

        Returns
        -------
        out : (batch, 1)  regression output
        """
        B = x.shape[0]

        # Initialise all states to |0...0⟩
        psi = torch.zeros(B, self.dim, dtype=torch.complex64, device=x.device)
        psi[:, 0] = 1.0   # |00⟩ = basis vector 0

        # Apply layers
        for layer in self.layers:
            psi = layer(psi, x)

        # Measure ⟨Z₀⟩ = Σ_s Z0[s] |ψ_s|²
        probs    = psi.abs().pow(2)                       # (B, dim)
        exp_Z0   = (probs * self.Z0.unsqueeze(0)).sum(1)  # (B,)

        return (self.scale * exp_Z0 + self.bias).unsqueeze(1)  # (B, 1)


qnn_model = QNN(n_qubits=2, n_layers=4).to(device)
n_qnn     = sum(p.numel() for p in qnn_model.parameters() if p.requires_grad)
print(f"\nQNN trainable parameters: {n_qnn:,}")
print(f"  n_qubits = {qnn_model.n_qubits},  n_layers = {len(qnn_model.layers)}")
print(f"  State dimension = {qnn_model.dim}  (= 2^{qnn_model.n_qubits})")
print(f"  Parameters: {len(qnn_model.layers)} layers × "
      f"{qnn_model.n_qubits} qubits × 4 angles + 2 output = {n_qnn}")

# =============================================================================
# §5 — SHARED TRAINING LOOP
# =============================================================================

def train(model: nn.Module,
          X_tr: torch.Tensor,
          Y_tr: torch.Tensor,
          X_va: torch.Tensor,
          Y_va: torch.Tensor,
          epochs: int = 500,
          lr: float = 1e-3,
          batch_size: int = 32,
          label: str = '') -> dict:
    """
    Adam + cosine annealing LR + gradient clipping.

    Both models share this function for a fair comparison.
    The QNN uses a smaller batch size (32) because its forward pass
    involves a batch of (2^n × 2^n) matrix multiplications — more
    expensive per sample than the classical NN.

    Returns
    -------
    history : dict with 'train_mse', 'val_mse', 'wall_s'
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    n         = X_tr.shape[0]
    history   = {'train_mse': [], 'val_mse': []}
    t0        = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm     = torch.randperm(n, device=device)
        ep_loss  = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            bi   = perm[start:start + batch_size]
            pred = model(X_tr[bi])
            loss = criterion(pred, Y_tr[bi])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1
        scheduler.step()
        history['train_mse'].append(ep_loss / n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            val_mse = criterion(model(X_va), Y_va).item()
        history['val_mse'].append(val_mse)

        if ep % 100 == 0 or ep == 1:
            print(f"  [{label}] ep {ep:4d}/{epochs}  "
                  f"train MSE={history['train_mse'][-1]:.5f}  "
                  f"val MSE={val_mse:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    history['wall_s'] = elapsed
    print(f"  Wall time: {elapsed:.1f}s")
    return history

# =============================================================================
# §6 — TRAINING
# =============================================================================

print(f"\n{SEP}")
print("TRAINING CLASSICAL NN")
print(SEP)
classical_history = train(
    classical_model, X_tr, Y_tr, X_va, Y_va,
    epochs=500, lr=1e-3, batch_size=32, label='ClassNN'
)

print(f"\n{SEP}")
print("TRAINING QUANTUM NN")
print(SEP)
print("  Note: forward pass simulates a quantum circuit via matrix algebra.")
print("  Gradients flow through rotation angles via torch.autograd.")
qnn_history = train(
    qnn_model, X_tr, Y_tr, X_va, Y_va,
    epochs=500, lr=1e-2, batch_size=32, label='QNN'
)

# =============================================================================
# §7 — EVALUATION
# =============================================================================

print(f"\n{SEP}")
print("EVALUATION ON TEST SET")
print(SEP)

criterion = nn.MSELoss()
classical_model.eval()
qnn_model.eval()

with torch.no_grad():
    class_pred_te = classical_model(X_te).cpu().numpy().flatten()
    qnn_pred_te   = qnn_model(X_te).cpu().numpy().flatten()
    y_te_np       = Y_te.cpu().numpy().flatten()

    class_pred_plot = classical_model(X_plot_t).cpu().numpy().flatten()
    qnn_pred_plot   = qnn_model(X_plot_t).cpu().numpy().flatten()

def metrics(pred, true):
    mse  = float(np.mean((pred - true)**2))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = float(1 - np.sum((pred-true)**2) / np.sum((true-true.mean())**2))
    return dict(MSE=mse, RMSE=math.sqrt(mse), MAE=mae, R2=r2)

cm = metrics(class_pred_te, y_te_np)
qm = metrics(qnn_pred_te,   y_te_np)

print(f"\n{'Model':<22s}  {'MSE':>8s}  {'RMSE':>8s}  {'MAE':>8s}  "
      f"{'R²':>6s}  {'Params':>8s}  {'Time':>8s}")
print("-" * 80)
for name, m, n_par, t in [
        ('Classical NN', cm, n_classical, classical_history['wall_s']),
        ('QNN (n=2, L=4)', qm, n_qnn,   qnn_history['wall_s'])]:
    print(f"{name:<22s}  {m['MSE']:>8.5f}  {m['RMSE']:>8.5f}  "
          f"{m['MAE']:>8.5f}  {m['R2']:>6.3f}  {n_par:>8,}  {t:>7.1f}s")

# =============================================================================
# §8 — VISUALISATION
# =============================================================================

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Classical Neural Network vs Quantum Neural Network\n"
    r"Regression on $f(x) = x^3 - 2x^2 + 0.5x + 1$",
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

# ── Panel 1: Fitted curves ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.scatter(x_tr, y_tr, s=12, alpha=0.35, color='steelblue',
           label=f'Train ({N_TRAIN})', zorder=2)
ax.scatter(x_te, y_te_np, s=16, alpha=0.5, color='darkorange',
           marker='D', label=f'Test ({N_TEST})', zorder=2)
ax.plot(x_plot, y_exact,         'k-',  lw=2.5, label='True f(x)', zorder=3)
ax.plot(x_plot, class_pred_plot, 'b--', lw=2.0,
        label=f"Classical NN (MAE={cm['MAE']:.4f})", zorder=4)
ax.plot(x_plot, qnn_pred_plot,   'r-',  lw=2.0,
        label=f"QNN (MAE={qm['MAE']:.4f})", zorder=4)
ax.set(xlabel='x', ylabel='f(x)',
       title='Fitted Curves vs True Polynomial')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Panel 2: Training loss curves ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ep = range(1, len(classical_history['train_mse']) + 1)
ax.semilogy(ep, classical_history['train_mse'], 'b-',  lw=2,
            label='Classical train')
ax.semilogy(ep, classical_history['val_mse'],   'b--', lw=1.5, alpha=0.8,
            label='Classical val')
ax.semilogy(ep, qnn_history['train_mse'],       'r-',  lw=2,
            label='QNN train')
ax.semilogy(ep, qnn_history['val_mse'],         'r--', lw=1.5, alpha=0.8,
            label='QNN val')
ax.set(xlabel='Epoch', ylabel='MSE (log scale)', title='Training Curves')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Panel 3: Residuals ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.axhline(0, color='k', lw=1.2, ls='--')
ax.scatter(y_te_np, class_pred_te - y_te_np, s=18, alpha=0.6, color='steelblue',
           label=f"Classical (RMSE={cm['RMSE']:.4f})")
ax.scatter(y_te_np, qnn_pred_te   - y_te_np, s=18, alpha=0.6, color='firebrick',
           marker='^', label=f"QNN (RMSE={qm['RMSE']:.4f})")
ax.set(xlabel='True f(x)', ylabel='Predicted − True',
       title='Residuals on Test Set')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Panel 4: Architecture and summary ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
summary = (
    "REGRESSION TARGET\n"
    "  f(x) = x³ - 2x² + 0.5x + 1\n"
    "  noise σ = 0.15,  x ∈ [-2, 2]\n\n"
    "CLASSICAL NN\n"
    "  1 → 32 → 32 → 16 → 1\n"
    "  Activation: Tanh\n"
    f"  Parameters: {n_classical:,}\n\n"
    "QUANTUM NN  (data re-uploading)\n"
    f"  n_qubits = {qnn_model.n_qubits},  n_layers = {len(qnn_model.layers)}\n"
    f"  State dim = {qnn_model.dim}  (= 2^n)\n"
    "  Each layer:\n"
    "    Encode: R_y(w·x) per qubit\n"
    "    Rotate: R_x R_y R_z per qubit\n"
    "    Entangle: CNOT ring\n"
    "  Output: scale·⟨Z₀⟩ + bias\n"
    f"  Parameters: {n_qnn:,}\n\n"
    "TRAINING\n"
    "  Optimiser: Adam + cosine LR\n"
    "  Loss: MSE,  epochs: 500\n"
    f"  Classical: {classical_history['wall_s']:.1f}s\n"
    f"  QNN:       {qnn_history['wall_s']:.1f}s\n\n"
    f"TEST RESULTS\n"
    f"  Classical R² = {cm['R2']:.4f}\n"
    f"  QNN       R² = {qm['R2']:.4f}"
)
ax.text(0.04, 0.97, summary, transform=ax.transAxes,
        fontsize=9.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('Architecture & Results', fontweight='bold', fontsize=10)

plt.savefig('qnn_vs_nn.png', dpi=140, bbox_inches='tight')
plt.show()
print("\n✓ Figure saved to qnn_vs_nn.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{SEP}")
print("FINAL SUMMARY")
print(SEP)
print(f"""
Task: regress  f(x) = x³ - 2x² + 0.5x + 1  from noisy samples.

Classical NN   1 → 32 → 32 → 16 → 1  (Tanh)
  MSE  = {cm['MSE']:.5f}    RMSE = {cm['RMSE']:.5f}
  MAE  = {cm['MAE']:.5f}    R²   = {cm['R2']:.4f}
  Params = {n_classical:,}   Time = {classical_history['wall_s']:.1f}s

QNN   2 qubits, 4 layers, data re-uploading
  MSE  = {qm['MSE']:.5f}    RMSE = {qm['RMSE']:.5f}
  MAE  = {qm['MAE']:.5f}    R²   = {qm['R2']:.4f}
  Params = {n_qnn:,}    Time = {qnn_history['wall_s']:.1f}s

Notes on the QNN implementation
--------------------------------
• Pure PyTorch — every gate is an explicit matrix multiply.
• No quantum library (no PennyLane, Qiskit, Cirq) is used.
• Gradients flow through cos/sin/exp in the gate matrices via autograd.
• The CNOT is a constant matrix; only the rotation angles are parameters.
• Data re-uploading (input at every layer) makes a 2-qubit circuit
  a universal approximator for 1-D functions via a truncated Fourier series.
• Forward pass cost scales as O(batch × n_layers × n_qubits × 2^n_qubits × 2^n_qubits),
  so it is slower per epoch than the classical NN on CPU.
• On actual quantum hardware, each layer corresponds to a hardware gate
  sequence; the variational parameters are the rotation angles trained here.
""")
