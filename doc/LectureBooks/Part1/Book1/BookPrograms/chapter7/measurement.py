"""Chapter 7, Sec. 7.4: measuring the energy on a quantum computer.

* the rotation recipe of Table 7.rotationcheck: <P> of a random two-qubit
  state from the measurement circuits of chapter 4 (nothing / H / S^dagger H
  before the read-out) against the matrix element;
* shot noise, Table 7.shots and Fig. 7.shotnoise: the standard error of the
  sampled energy of a fixed trial state of the two-mode Hamiltonian against
  the number of shots, and the 1/sqrt(N) law;
* the marginal trap, Theorem 7.marginal: the two Bell states with identical
  single-qubit statistics, the flat and the fictitious energy landscape of
  Fig. 7.fakelandscape, and a VQE run with marginal read-out on the two-mode
  Hamiltonian, Table 7.marginal.

About ten seconds (the repeated shot experiments dominate).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from pauli import PauliSum, string_matrix
from statevector import StateVector
from pauli_circuits import measure_group, energy_from_shots
from vqe import VQE, HardwareEfficient, PauliRotationAnsatz, run_restarts
from bookplot import savefig, plt


def marginal_energy(H, psi):
    """The *wrong* reconstruction: every string replaced by the product of its
    single-qubit expectation values, Eq. (7.marginalwrong)."""
    sv = StateVector(H.n, psi)
    singles = {}
    for q in range(H.n):
        for c in "XYZ":
            singles[(q, c)] = sv.expectation(G.PAULIS[c], q)
    E = 0.0
    for label, h in H.terms.items():
        prod = 1.0
        for q, c in enumerate(label):
            if c != "I":
                prod *= singles[(q, c)]
        E += np.real(h) * prod
    return E


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    rng = np.random.default_rng(20)

    # ------------------------------------------------ the rotation recipe (Table 7.rotationcheck)
    print("=== The rotation recipe on a random two-qubit state (Table 7.rotationcheck)")
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    sv = StateVector(2, psi)
    print(f"{'string':>7s} {'rotation + read-out':>20s} {'<psi|P|psi>':>12s} {'difference':>11s}")
    for label in ("ZI", "XI", "YI", "XX", "YY", "ZZ", "XZ", "YX"):
        # exact outcome distribution instead of finite shots: average the parity over |amplitude|^2
        rotated = sv.copy()
        for q, c in enumerate(label):
            if c == "X":
                rotated.h(q)
            elif c == "Y":
                rotated.apply(G.S.conj().T, q); rotated.h(q)
        p = rotated.probabilities()
        support = [q for q, c in enumerate(label) if c != "I"]
        est = sum(pk * (-1) ** sum(int(format(k, "02b")[q]) for q in support) for k, pk in enumerate(p))
        exact = sv.expectation(string_matrix(label))
        print(f"{label:>7s} {est:20.6f} {exact:12.6f} {abs(est - exact):11.0e}")
        assert abs(est - exact) < 1e-12

    # ------------------------------------------------ shot noise (Table 7.shots, Fig. 7.shotnoise)
    print("\n=== Shot noise in the energy of a fixed trial state of the two-mode Hamiltonian (Table 7.shots)")
    H2 = PauliSum({"II": 1.625, "ZI": -0.625, "IZ": -1.125, "ZZ": 0.125, "XX": 0.15, "YY": 0.15})
    hea = HardwareEfficient(2, 2)
    theta = hea.random_parameters(rng, 1.0)
    state = StateVector(2, hea.state(theta))
    E_exact = float(np.real(np.vdot(state.psi, H2.matrix() @ state.psi)))
    Lambda = sum(abs(c) for k, c in H2.terms.items() if k != "II")
    print(f"  exact energy {E_exact:.6f}, Pauli 1-norm of the traceless part {Lambda:.3f}, {len(H2.commuting_groups())} measurement settings")
    print(f"{'shots per setting':>18s} {'standard error':>15s} {'x sqrt(N)':>10s} {'bias':>8s}")
    rows = []
    for N in (10, 100, 1000, 10000):
        Es = np.array([energy_from_shots(H2, state, N, rng)[0] for _ in range(200)])
        rows.append((N, Es.std(), Es.std() * np.sqrt(N), Es.mean() - E_exact))
        print(f"{N:18d} {Es.std():15.4f} {Es.std() * np.sqrt(N):10.2f} {Es.mean() - E_exact:+8.4f}")
    assert all(0.6 < r[2] / rows[0][2] < 1.6 for r in rows)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    Ns = [r[0] for r in rows]
    ax.loglog(Ns, [r[1] for r in rows], "o-", label="measured spread of 200 repetitions")
    ax.loglog(Ns, [rows[0][1] * np.sqrt(rows[0][0] / N) for N in Ns], "k--", lw=0.8, label=r"$N^{-1/2}$ through the first point")
    ax.set_xlabel("shots per measurement setting $N$"); ax.set_ylabel("standard error of the energy"); ax.legend(fontsize=8)
    savefig(fig, 7, "shotnoise")

    # ------------------------------------------------ the marginal trap
    print("\n=== Marginals do not determine correlators (Theorem 7.marginal)")
    phi_plus = StateVector(2, [1, 0, 0, 1]); psi_plus = StateVector(2, [0, 1, 1, 0])
    for name, st in (("Phi+", phi_plus), ("Psi+", psi_plus)):
        singles = [st.expectation(G.PAULIS[c], q) for q in range(2) for c in "XYZ"]
        zz = st.expectation(string_matrix("ZZ"))
        print(f"  |{name}>: <X_q>, <Y_q>, <Z_q> = {np.round(singles, 12)}, <Z Z> = {zz:+.0f}")
        assert np.allclose(singles, 0)
    thetas = np.linspace(0, np.pi, 181)
    true = np.ones_like(thetas)
    naive = np.cos(2 * thetas) ** 2
    ZZ = PauliSum({"ZZ": 1.0})
    for th in (0.3, 1.1):
        psi = np.array([np.cos(th), 0, 0, np.sin(th)])
        assert abs(marginal_energy(ZZ, psi) - np.cos(2 * th) ** 2) < 1e-12
        assert abs(np.real(np.vdot(psi, ZZ.matrix() @ psi)) - 1) < 1e-12
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(thetas / np.pi, true, "k-", label=r"$E_{\rm true}(\theta)=1$")
    ax.plot(thetas / np.pi, naive, "--", label=r"$E_{\rm naive}(\theta)=\cos^2 2\theta$")
    ax.plot([0.25, 0.75], [0, 0], "o", mfc="none", ms=8, color="C3")
    ax.set_xlabel(r"$\theta/\pi$"); ax.set_ylabel("energy"); ax.legend(fontsize=8)
    savefig(fig, 7, "fakelandscape")

    print("\n=== A VQE run with marginal read-out on the two-level pairing Hamiltonian, hardware-efficient ansatz (Table 7.marginal)")
    Hp = PauliSum({"II": 0.5, "ZI": 0.25, "IZ": -0.75, "XX": -0.25, "YY": -0.25})      # Eq. (4.jwpairingtwolevels)
    H2, E0 = Hp, np.linalg.eigvalsh(Hp.matrix())[0]
    hea = HardwareEfficient(2, 2)
    theta0 = hea.random_parameters(np.random.default_rng(3), 1.0)
    vqe = VQE(H2, hea)
    th, E_joint = vqe.bfgs(theta0)
    vqe_m = VQE(H2, hea)
    vqe_m.energy = lambda theta: marginal_energy(H2, hea.state(theta))
    th_m, E_marg = vqe_m.bfgs(theta0)
    E_true_of_marg = vqe.exact_energy(th_m)
    print(f"  exact ground state                       {E0:.6f}")
    print(f"  VQE, joint read-out                      {E_joint:.6f}")
    print(f"  VQE, marginal reconstruction (reported)  {E_marg:.6f}")
    print(f"  true energy of the state it prepared     {E_true_of_marg:.6f}")
    assert abs(E_joint - E0) < 1e-6 and abs(E_marg - E_true_of_marg) > 1e-3 and E_true_of_marg > E0 + 1e-2
    print("\nmeasurement checks passed")
