import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit
import matplotlib.animation as animation

# -------------------- PARAMETERS --------------------
n_qubits = 3
gamma = 1.0
B0_true = 0.8
omega_true = 1.0
phi0_true = 0.0
T2 = 3.0
shots = 100
sigma_noise = 0.05
times = np.linspace(0.1, 5, 100)

# -------------------- FUNCTIONS --------------------
def B_field(t, B0, omega):
    return B0 * np.sin(omega * t)

def ideal_expectation(t, B0, omega, phi0):
    phase = gamma * n_qubits * B0 * np.sin(omega * t) * t + phi0
    return np.cos(phase)

def decohered_expectation(t, B0, omega, phi0, T2):
    return ideal_expectation(t, B0, omega, phi0) * np.exp(-t / T2)

def noisy_measurement(expectation, shots, sigma):
    p = (1 + expectation) / 2
    counts = np.random.binomial(shots, p)
    mean_measurement = 2 * (counts / shots) - 1
    return mean_measurement + np.random.normal(0, sigma)

def fit_model(t, B0, omega, phi0):
    phase = gamma * n_qubits * B0 * np.sin(omega * t) * t + phi0
    return np.cos(phase) * np.exp(-t / T2)

# -------------------- SIMULATE DATA --------------------
true_expectations = []
measured_expectations = []
for t in times:
    exp_t = decohered_expectation(t, B0_true, omega_true, phi0_true, T2)
    meas_t = noisy_measurement(exp_t, shots, sigma_noise)
    true_expectations.append(exp_t)
    measured_expectations.append(meas_t)

# -------------------- CURVE FIT (MAP ESTIMATE) --------------------
initial_guess = [0.5, 1.2, 0.1]
params_opt, _ = curve_fit(fit_model, times, measured_expectations, p0=initial_guess)
B0_map, omega_map, phi0_map = params_opt

# -------------------- INTERACTIVE FIT VIEWER --------------------
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.3)
ax.scatter(times, measured_expectations, label='Noisy Measurements', color='red', s=10)
ax.plot(times, true_expectations, label='True Signal', linewidth=2, alpha=0.5)
model_line, = ax.plot(times, fit_model(times, B0_map, omega_map, phi0_map),
                      label='Model Fit', color='green', linewidth=2)
ax.set_title("Interactive Quantum Sensor Fit")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\langle X^{\otimes n} \rangle$")
ax.legend()
ax.grid(True)

# Sliders
axcolor = 'lightgoldenrodyellow'
ax_B0 = plt.axes([0.15, 0.2, 0.7, 0.03], facecolor=axcolor)
ax_omega = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
ax_phi0 = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=axcolor)

slider_B0 = Slider(ax_B0, 'B₀', 0.3, 1.2, valinit=B0_map)
slider_omega = Slider(ax_omega, 'ω', 0.5, 1.5, valinit=omega_map)
slider_phi0 = Slider(ax_phi0, 'φ₀', -np.pi, np.pi, valinit=phi0_map)

def update(val):
    B0 = slider_B0.val
    omega = slider_omega.val
    phi0 = slider_phi0.val
    model_line.set_ydata(fit_model(times, B0, omega, phi0))
    fig.canvas.draw_idle()

slider_B0.on_changed(update)
slider_omega.on_changed(update)
slider_phi0.on_changed(update)

plt.show()

# -------------------- TRUE PHASE ANIMATION --------------------
true_phase = gamma * n_qubits * B_field(times, B0_true, omega_true) * times + phi0_true
wrapped_phase = np.mod(true_phase + np.pi, 2 * np.pi) - np.pi  # wrap to [-π, π]

fig_phase, ax_phase = plt.subplots(figsize=(8, 4))
ax_phase.set_title("True Quantum Phase Evolution")
ax_phase.set_xlabel("Time [s]")
ax_phase.set_ylabel("Phase [rad]")
ax_phase.set_ylim([-np.pi, np.pi])
ax_phase.grid(True)

(line_phase,) = ax_phase.plot([], [], lw=2, label='True Phase')
(time_dot,) = ax_phase.plot([], [], 'ro')
ax_phase.legend()

def init():
    line_phase.set_data([], [])
    time_dot.set_data([], [])
    return line_phase, time_dot

def animate(i):
    t = times[:i]
    phi = wrapped_phase[:i]
    line_phase.set_data(t, phi)
    if i > 0:
        time_dot.set_data(t[-1], phi[-1])
    return line_phase, time_dot

ani = animation.FuncAnimation(
    fig_phase, animate, frames=len(times), init_func=init,
    blit=True, interval=100, repeat=False
)

plt.show()
