import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Quantum Gates and Helpers ---

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

def kron_n(*ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def GHZ_state(n):
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    state[-1] = 1
    return state / np.sqrt(2)

def expm(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.conj().T

def evolve_magnetic(state, B_t, t, gamma, n):
    phi = gamma * B_t * t
    U = np.eye(1, dtype=complex)
    for _ in range(n):
        U = np.kron(U, expm(-1j * phi * Z / 2))
    return U @ state

def measure_in_X_basis(state, n):
    Xn = kron_n(*([X] * n))
    return np.real(np.vdot(state, Xn @ state))

# --- Sensing Parameters ---

n = 3            # number of entangled qubits
gamma = 1.0      # gyromagnetic ratio
true_B0 = 0.8    # true unknown amplitude
omega = 1.0      # known frequency
times = np.linspace(0.1, 5, 200)  # sensing times

# --- Simulate Sensor Signal ---

def B_field(t, B0): return B0 * np.sin(omega * t)

expectations = []
for t in times:
    Bt = B_field(t, true_B0)
    psi = GHZ_state(n)
    psi_t = evolve_magnetic(psi, Bt, t, gamma, n)
    expectations.append(measure_in_X_basis(psi_t, n))

expectations = np.array(expectations)

# --- Step 1: Optimal Measurement Time ---

grad = np.gradient(expectations, times)
optimal_time = times[np.argmax(np.abs(grad))]

print(f"🧠 Optimal measurement time: t = {optimal_time:.3f} s (max |d⟨Xⁿ⟩/dt|)")

# --- Step 2: Estimate Unknown B₀ ---

# Fit model: cos(γ n B₀ sin(ω t) * t)
def model(t, B0_est):
    return np.cos(gamma * n * B0_est * np.sin(omega * t) * t)

B0_fit, _ = curve_fit(model, times, expectations, p0=[0.5])
print(f"🔍 Estimated B₀: {B0_fit[0]:.3f} (true: {true_B0})")

plt.figure(figsize=(10, 5))
plt.plot(times, expectations, label="Quantum Sensor Signal", color='blue')
plt.plot(times, model(times, B0_fit[0]), '--', label=f"Fitted Model (B₀ ≈ {B0_fit[0]:.3f})", color='red')
plt.axvline(optimal_time, color='green', linestyle=':', label=f"Optimal t = {optimal_time:.2f}")
plt.xlabel("Time [s]")
plt.ylabel(r"$\langle X^{\otimes n} \rangle$")
plt.title(f"Quantum Sensor with GHZ({n}) State — Estimating $B_0$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
