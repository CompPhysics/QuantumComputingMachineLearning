"""Chapter 2, Sec. 2.4: projective measurements.

Born-rule probabilities and post-measurement states for a single qubit
(Eqs. 2.bornrule-2.collapse), the trace form p = Tr(P rho) (Eq. 2.borntrace),
the marginal probability of one qubit in an n-qubit register
(Eq. 2.marginal), measurement in the X basis through a Hadamard
(Eq. 2.xbasis), and the statistics of repeated measurements: Fig. 2.shots
shows how the estimate of |alpha|^2 from N shots converges as 1/sqrt(N).
Runs in a few seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from statevector import StateVector
from bookplot import savefig, plt


def projectors(n_qubits, qubit):
    """P_0 and P_1 for measuring `qubit` of an n-qubit register."""
    P = []
    for k in (0, 1):
        ops = [G.I] * n_qubits
        ops[qubit] = np.diag([1 - k, k]).astype(complex)
        P.append(G.kron(*ops))
    return P


def born_rule(psi, P):
    """p = <psi|P|psi> and the collapsed state P|psi>/sqrt(p)."""
    p = float(np.real(np.vdot(psi, P @ psi)))
    post = P @ psi / np.sqrt(p) if p > 1e-15 else None
    return p, post


def shots_experiment(psi, shots_list, rng):
    """Estimate P(0) from N single-shot measurements for each N in shots_list."""
    p0 = abs(psi[0]) ** 2
    est = []
    for N in shots_list:
        outcomes = rng.random(N) < (1 - p0)          # True means outcome 1
        est.append(1 - outcomes.mean())
    return np.array(est)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(2026)
    # --- single qubit, Eqs. (2.bornrule)-(2.collapse)
    alpha, beta = np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.4j)
    psi = np.array([alpha, beta])
    P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
    for k, P in enumerate((P0, P1)):
        p, post = born_rule(psi, P)
        print(f"outcome {k}: probability {p:.4f}, post-measurement state {post}")
    # --- the trace form, Eq. (2.borntrace)
    rho = np.outer(psi, psi.conj())
    assert abs(np.trace(P0 @ rho).real - 0.3) < 1e-12
    print("Tr(P_0 rho) =", np.trace(P0 @ rho).real)
    # --- expectation value of Z and of X
    print("<Z> = p0 - p1 =", np.vdot(psi, G.Z @ psi).real, "  <X> =", np.vdot(psi, G.X @ psi).real)
    # --- X-basis measurement through a Hadamard, Eq. (2.xbasis)
    plus = np.array([1, 1]) / np.sqrt(2)
    p_plus = abs(np.vdot(plus, psi)) ** 2
    p_h0 = abs((G.H @ psi)[0]) ** 2
    assert abs(p_plus - p_h0) < 1e-12
    print(f"P(+) = |<+|psi>|^2 = {p_plus:.4f} = P(0) after H = {p_h0:.4f}")
    # --- one qubit of a three-qubit register, Eq. (2.marginal)
    sv = StateVector(3).h(0).cnot(0, 1).ry(1.1, 2)
    P = projectors(3, 1)
    p_direct = [born_rule(sv.psi, Pk)[0] for Pk in P]
    assert np.allclose(p_direct, sv.marginal(1))
    print("P(qubit 1 = 0), P(qubit 1 = 1) in |Psi> =", sv.marginal(1))
    post = sv.copy(); out, p = post.measure(1, outcome=1)
    print("after finding qubit 1 = 1 the register is", post)
    # --- Fig. 2.shots: statistics of repeated measurements
    shots = np.unique(np.logspace(0, 5, 40).astype(int))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for i in range(5):
        est = shots_experiment(psi, shots, rng)
        ax.plot(shots, est, "-", lw=0.9, alpha=0.8, label="independent runs" if i == 0 else None)
    ax.axhline(0.3, color="k", ls="--", lw=1, label=r"$|\alpha|^2 = 0.3$")
    err = np.sqrt(0.3 * 0.7 / shots)
    ax.fill_between(shots, 0.3 - err, 0.3 + err, color="gray", alpha=0.25, label=r"$\pm\sqrt{p(1-p)/N}$")
    ax.set_xscale("log"); ax.set_xlabel("number of shots $N$"); ax.set_ylabel("estimate of $P(0)$")
    ax.set_ylim(0, 0.8); ax.legend(loc="upper right")
    print("figure written to", savefig(fig, 2, "measurement_shots"))
