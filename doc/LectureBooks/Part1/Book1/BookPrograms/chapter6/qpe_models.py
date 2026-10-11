"""Chapter 6, Secs. 6.6-6.8: phase estimation on the model Hamiltonians of chapter 4.

* the two small Hamiltonians worked through by hand: the tapered one-qubit
  pairing Hamiltonian of Eq. (4.tapered) and the two-level pairing
  Hamiltonian of Eq. (4.jwpairingtwolevels) -- Tables 6.onequbit, 6.twoqubit;
* the four models in the encodings of Table 4.summary -- Lipkin (one qubit
  per doublet), pairing L = 4 (one qubit per level), the Hubbard ring on eight
  qubits, the Heisenberg ring -- with their Pauli-sum bookkeeping, the choice
  of the evolution time, and phase estimation of the ground-state energy
  against the size of the control register -- Tables 6.models, 6.qpemodels,
  Fig. 6.modelerrors;
* the Trotter error inside phase estimation, on the Heisenberg ring --
  Table 6.trotter, Fig. 6.trotterbudget;
* trial states that are not eigenstates: overlaps of the reference
  determinants, the Lipkin mean field, spectroscopy from the histogram and
  phase estimation as state preparation -- Tables 6.overlaps, 6.spectroscopy,
  Fig. 6.spectroscopy.

Every energy is checked against the exact diagonalisation of chapters 3
and 4.  About twenty seconds (the eight-qubit Hubbard runs dominate).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from pauli import PauliSum
from jordanwigner import JordanWigner
from fockspace import FockSpace
from models import LipkinModel, PairingModel, HubbardModel, HeisenbergModel
from mapped_models import lipkin_level_encoding
from pauli_circuits import compact_pairing, trotter_step, gate_counts
from matrixfunctions import expm_hermitian
from statevector import StateVector
from phase_estimation import PhaseEstimation, choose_time, phase_to_energy, signed_phase
from bookplot import savefig, plt


# ------------------------------------------------------------ Hamiltonians
def split_identity(H):
    """(h_I, H - h_I I): the identity coefficient and the traceless part of a PauliSum."""
    hI = H.terms.get("I" * H.n, 0.0)
    rest = PauliSum({k: c for k, c in H.terms.items() if k != "I" * H.n}, H.n)
    return float(np.real(hI)), rest


def hubbard_ring_spin_major(n_sites=4, t=1.0, U=4.0):
    """The Hubbard ring on 2 n_sites qubits, all spin-up modes first (Sec. 4.4.4)."""
    hub = HubbardModel(n_sites, t=t, U=U)
    perm = [2 * i for i in range(n_sites)] + [2 * i + 1 for i in range(n_sites)]
    h = hub.h[np.ix_(perm, perm)]
    v = hub.v[np.ix_(perm, perm, perm, perm)]
    return hub, JordanWigner(2 * n_sites).map_hamiltonian(h, v)


def sector_ground_state(H, n_ones):
    """Lowest eigenpair of a PauliSum in the sector with n_ones qubits set, embedded in 2^n."""
    idx = FockSpace(H.n).sector(n_ones)
    M = H.matrix()[np.ix_(idx, idx)]
    w, V = np.linalg.eigh(M)
    psi = np.zeros(2 ** H.n, dtype=complex)
    psi[idx] = V[:, 0]
    return w[0], psi, w


def book_models():
    """The four Hamiltonians of Table 4.summary with their exact ground states."""
    out = []
    lip = LipkinModel(4, eps=2.0, V=-1 / 3, W=-1 / 4)
    H = lipkin_level_encoding(lip)
    w, V = np.linalg.eigh(H.matrix())
    out.append(("Lipkin, $N=4$", H, w[0], V[:, 0], None))
    pm = PairingModel(4, g=1.0)
    H = compact_pairing(pm)
    E0, psi, _ = sector_ground_state(H, 2)
    out.append(("pairing, $L=4$", H, E0, psi, 2))
    hub, H = hubbard_ring_spin_major()
    E0, psi, _ = sector_ground_state(H, 4)
    out.append(("Hubbard ring", H, E0, psi, 4))
    heis = HeisenbergModel(4)
    H = PauliSum.from_matrix(heis.full_matrix())
    w, V = np.linalg.eigh(H.matrix())
    out.append(("Heisenberg ring", H, w[0], V[:, 0], None))
    return out


def run_qpe(H, psi, m, s=0.9, method=None, U=None):
    """Phase estimation of the PauliSum H on the state psi with m control qubits.

    The identity is removed and added back; t = pi s / Lambda.  Returns the
    PhaseEstimation object with the attributes hI, t and E (the energy grid)."""
    hI, Ht = split_identity(H)
    t = choose_time(H, s)
    method = method or ("spectral" if H.n >= 6 else "circuit")
    if U is None:
        qpe = PhaseEstimation.from_hamiltonian(Ht.matrix(), t, m, method=method)
    else:
        qpe = PhaseEstimation.from_unitary(U, m, method="circuit")
        qpe.t = t
    qpe.run(psi)
    qpe.hI = hI
    qpe.E = phase_to_energy(np.arange(2 ** m), m, t) + hI
    return qpe


def qpe_table(H, psi, E0, ms, **kw):
    rows = []
    for m in ms:
        q = run_qpe(H, psi, m, **kw)
        k, phi, p = q.estimate()
        rows.append((m, k, q.E[k], abs(q.E[k] - E0), p, q.t))
    return rows


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # ============================================ the one-qubit pairing Hamiltonian
    print("=== One qubit: the tapered two-level pairing Hamiltonian H = I/2 + Z - X/2 (Eq. 4.tapered)")
    H1 = PauliSum({"I": 0.5, "Z": 1.0, "X": -0.5})
    w = np.linalg.eigvalsh(H1.matrix())
    Eexact = 0.5 - np.sqrt(5) / 2
    assert abs(w[0] - Eexact) < 1e-12
    hI, Ht = split_identity(H1)
    print(f"  spectrum {w}, Lambda = {Ht.norm_bound()} after removing the identity, t = {choose_time(H1):.4f}")
    _, V = np.linalg.eigh(H1.matrix())
    psi1 = V[:, 0]
    print(f"{'m':>3s} {'outcome':>12s} {'E':>10s} {'|E-E0|':>9s} {'prob':>7s}")
    rows1 = qpe_table(H1, psi1, Eexact, (3, 4, 5, 6, 7, 8, 10))
    for m, k, E, err, p, t in rows1:
        print(f"{m:3d} {format(k, f'0{m}b'):>12s} {E:10.6f} {err:9.1e} {p:7.3f}")
    assert all(err < np.pi / (t * 2 ** m) for m, k, E, err, p, t in rows1)
    # the phase register wraps if the identity is kept and t is chosen from the full 1-norm
    q = PhaseEstimation.from_hamiltonian(H1.matrix(), np.pi * 0.9 / H1.norm_bound(), 8); q.run(psi1)
    k, _, _ = q.estimate()
    print(f"  keeping the identity: t = {q.t:.4f}, E_QPE = {phase_to_energy(k, 8, q.t):.6f} (exact {Eexact:.6f}); "
          f"without the shift the resolution is {np.pi / (q.t * 2 ** 8):.4f} instead of {np.pi / (rows1[-2][5] * 2 ** 8):.4f}")

    # the untapered two-level pairing Hamiltonian on two qubits gives the same bits
    H2p = compact_pairing(PairingModel(2, g=1.0))
    assert H2p.terms == {"II": 0.5, "ZI": 0.25, "IZ": -0.75, "XX": -0.25, "YY": -0.25}
    E0p, psi2p, _ = sector_ground_state(H2p, 1)
    rows2p = qpe_table(H2p, psi2p, E0p, (3, 4, 5, 6, 7, 8, 10))
    assert all(a[1] == b[1] and abs(a[4] - b[4]) < 1e-10 for a, b in zip(rows1, rows2p))
    print("  the two-qubit form of Eq. (4.jwpairingtwolevels) in the one-pair sector returns the same outcomes and probabilities")

    # ============================================ the two-mode Hamiltonian on two qubits
    print("\n=== Two qubits: the two-mode Hamiltonian of Eq. (4.twoqubitH), eps = (1, 2), t = 0.3, U = 0.5")
    H2 = PauliSum({"II": 1.625, "ZI": -0.625, "IZ": -1.125, "ZZ": 0.125, "XX": 0.15, "YY": 0.15})
    w2 = np.linalg.eigvalsh(H2.matrix())
    E0_2, psi2, ws = sector_ground_state(H2, 1)
    assert abs(E0_2 - 0.916905) < 1e-6 and abs(ws[1] - 2.083095) < 1e-6
    hI2, Ht2 = split_identity(H2)
    print(f"  spectrum {w2}; the one-particle block holds {ws[0]:.6f} and {ws[1]:.6f}")
    print(f"  Lambda = {Ht2.norm_bound()}, t = {choose_time(H2):.4f}, groups of commuting strings: {Ht2.commuting_groups(qubitwise=False)}")
    rows2 = qpe_table(H2, psi2, E0_2, (3, 4, 5, 6, 7, 8, 10))
    print(f"{'m':>3s} {'outcome':>12s} {'E':>10s} {'|E-E0|':>9s} {'prob':>7s}")
    for m, k, E, err, p, t in rows2:
        print(f"{m:3d} {format(k, f'0{m}b'):>12s} {E:10.6f} {err:9.1e} {p:7.3f}")
    assert all(err < np.pi / (t * 2 ** m) for m, k, E, err, p, t in rows2)
    # the reference |10> (the particle in mode 0) as trial state
    trial = StateVector.from_bits("10").psi
    q = run_qpe(H2, trial, 8)
    p = q.distribution()
    c = np.abs(np.linalg.eigh(H2.matrix()[np.ix_(FockSpace(2).sector(1), FockSpace(2).sector(1))])[1].T @ trial[FockSpace(2).sector(1)]) ** 2
    weights = []
    for e in ws:                                  # weight within one grid step of each eigenvalue
        k = int(np.argmin(np.abs(q.E - e)))
        weights.append(p[k - 1:k + 2].sum())
    print(f"  trial state |10>: |c_j|^2 = {np.round(c, 4)} on the one-particle eigenvalues {ws}; "
          f"histogram weight within one grid step of each: {np.round(weights, 4)} (m = 8)")
    assert all(8 / np.pi ** 2 * cj - 1e-9 <= wj <= cj + 1e-9 for cj, wj in zip(c, weights))

    # ============================================ the four models
    print("\n=== The models of Table 4.summary as inputs to phase estimation (Table 6.models)")
    models = book_models()
    print(f"{'model':>16s} {'qubits':>6s} {'strings':>7s} {'weight':>6s} {'groups':>6s} {'Lambda':>7s} {'||H||':>7s} {'t':>7s} {'E0':>11s}")
    refs = {"Lipkin, $N=4$": -4.21288, "pairing, $L=4$": 0.63554847, "Hubbard ring": -2.10274848, "Heisenberg ring": -2.0}
    for name, H, E0, psi, sector in models:
        hI, Ht = split_identity(H)
        norm = np.abs(np.linalg.eigvalsh(Ht.matrix())).max()
        groups = len(Ht.commuting_groups(qubitwise=False))
        print(f"{name:>16s} {H.n:6d} {H.n_strings():7d} {H.max_weight():6d} {groups:6d} {Ht.norm_bound():7.3f} {norm:7.3f} {choose_time(H):7.4f} {E0:11.8f}")
        assert abs(E0 - refs[name]) < 1e-5

    print("\n=== Phase estimation of the ground-state energies, exact eigenstate and exact U (Table 6.qpemodels)")
    print(f"{'model':>16s} {'m':>3s} {'outcome':>13s} {'E_QPE':>11s} {'error':>8s} {'prob':>6s}")
    errors = {}
    for name, H, E0, psi, sector in models:
        rows = qpe_table(H, psi, E0, (4, 6, 8, 10))
        for m, k, E, err, p, t in rows:
            print(f"{name:>16s} {m:3d} {format(k, f'0{m}b'):>13s} {E:11.6f} {err:8.1e} {p:6.3f}")
            assert err <= np.pi / (t * 2 ** m) + 1e-12       # never further than half a grid step
            assert p > 4 / np.pi ** 2 - 1e-9
        # error against m for the figure, with the spectral method throughout
        ms = list(range(2, 13))
        errors[name] = [(m, r[3], np.pi / (r[5] * 2 ** m)) for m, r in
                        zip(ms, qpe_table(H, psi, E0, ms, method="spectral"))]
    print("\n=== What a tighter bound on the spectrum buys: the Hubbard ring with m = 10")
    name, H, E0, psi, sector = models[2]
    hI, Ht = split_identity(H)
    wsec = sector_ground_state(H, 4)[2]
    for label, t in (("Pauli 1-norm, t = 0.9 pi / Lambda", choose_time(H)),
                     ("spectral norm, t = 0.9 pi / ||H||", 0.9 * np.pi / np.abs(np.linalg.eigvalsh(Ht.matrix())).max()),
                     ("four-particle sector, t = 0.9 pi / max|E - h_I|", 0.9 * np.pi / np.abs(wsec - hI).max())):
        q = PhaseEstimation.from_hamiltonian(Ht.matrix(), t, 10, method="spectral"); q.run(psi)
        k, _, p = q.estimate()
        E = phase_to_energy(k, 10, t) + hI
        print(f"  {label:>48s}: t = {t:.4f}, grid step {2 * np.pi / (t * 2 ** 10):.4f}, E_QPE = {E:.6f}, error {abs(E - E0):.1e}, prob {p:.3f}")
    # the two implementations agree on a four-qubit model
    name, H, E0, psi, sector = models[0]
    a = run_qpe(H, psi, 6, method="circuit"); b = run_qpe(H, psi, 6, method="spectral")
    assert np.allclose(a.distribution(), b.distribution())
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for (name, rows), mk in zip(errors.items(), "osd^"):
        ax.semilogy([r[0] for r in rows], [max(r[1], 1e-16) for r in rows], mk + "-", ms=5, label=name.replace("$", ""))
    ax.semilogy([r[0] for r in errors["Heisenberg ring"]], [r[2] for r in errors["Heisenberg ring"]], "k--", lw=0.8, label=r"half a grid step, Heisenberg")
    ax.set_xlabel("control qubits $m$"); ax.set_ylabel(r"$|E_{\mathrm{QPE}}-E_0|$"); ax.legend(fontsize=8)
    savefig(fig, 6, "model_errors")

    # ============================================ Trotter error inside QPE
    print("\n=== Trotterised evolution inside phase estimation: the Heisenberg ring, m = 8 (Table 6.trotter)")
    name, H, E0, psi, sector = models[3]
    hI, Ht = split_identity(H)
    t = choose_time(H)
    Uexact = expm_hermitian(Ht.matrix(), -1j * t)
    print(f"  t = {t:.4f}; one first-order step: {gate_counts(trotter_step(Ht, t))} (gates, CNOTs, depth)")
    print(f"{'order':>5s} {'r':>3s} {'E_QPE':>10s} {'error':>8s} {'prob':>6s} {'||U_r - U||':>12s} {'CNOTs in U_r':>13s}")
    trot = {1: [], 2: []}
    for order in (1, 2):
        for r in (1, 2, 4, 8, 16, 32):
            step = trotter_step(Ht, t / r, order=order)
            Ustep = step.unitary()
            Ur = np.linalg.matrix_power(Ustep, r)
            q = run_qpe(H, psi, 8, U=Ur)
            k, _, p = q.estimate()
            operr = np.linalg.norm(Ur - Uexact, 2)
            cnots = gate_counts(step)[1] * r
            trot[order].append((r, q.E[k], abs(q.E[k] - E0), p, operr, cnots))
            print(f"{order:5d} {r:3d} {q.E[k]:10.6f} {abs(q.E[k] - E0):8.1e} {p:6.3f} {operr:12.2e} {cnots:13d}")
    # first order falls as 1/r, second order as 1/r^2
    o1 = [x[4] for x in trot[1]]; o2 = [x[4] for x in trot[2]]
    assert all(1.7 < a / b < 2.3 for a, b in zip(o1[1:-1], o1[2:]))
    assert all(3.5 < a / b < 4.5 for a, b in zip(o2[1:-1], o2[2:]))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.loglog([x[0] for x in trot[1]], [x[4] for x in trot[1]], "o-", label=r"$\Vert U_r-U\Vert$, first order")
    ax.loglog([x[0] for x in trot[2]], [x[4] for x in trot[2]], "o--", label=r"$\Vert U_r-U\Vert$, second order")
    ax.loglog([x[0] for x in trot[1]], [max(x[2], 1e-6) for x in trot[1]], "s-", label="energy error, first order")
    ax.loglog([x[0] for x in trot[2]], [max(x[2], 1e-6) for x in trot[2]], "s--", label="energy error, second order")
    ax.axhline(np.pi / (t * 2 ** 8), color="k", ls=":", lw=0.8, label="half a grid step, $m=8$")
    ax.set_xlabel("Trotter steps $r$"); ax.legend(fontsize=8)
    savefig(fig, 6, "trotter_budget")

    # ============================================ trial states and overlaps
    print("\n=== Overlaps of simple trial states with the exact ground states (Table 6.overlaps)")
    trials = {"Lipkin, $N=4$": ("0000", "reference, all doublets down"),
              "pairing, $L=4$": ("1100", "reference, two pairs in the lowest levels"),
              "Hubbard ring": ("10100101", "N\\'eel determinant"),
              "Heisenberg ring": ("0101", "N\\'eel state")}
    print(f"{'model':>16s} {'trial state':>12s} {'|c_0|^2':>8s} {'1/|c_0|^2':>9s} {'P(E_0 bits), m=8':>17s}")
    for name, H, E0, psi, sector in models:
        bits, desc = trials[name]
        trial = StateVector.from_bits(bits).psi
        c0 = abs(np.vdot(psi, trial)) ** 2
        q = run_qpe(H, trial, 8)
        p = q.distribution()
        k0 = int(np.argmin(np.abs(q.E - E0)))
        print(f"{name:>16s} {bits:>12s} {c0:8.4f} {1 / c0:9.1f} {p[k0]:17.4f}")

    print("\n=== The Lipkin model at chi = 2 (eps = 1, V = -2/3, W = 0): reference, mean field and exact (Fig. 6.spectroscopy)")
    lip = LipkinModel(4, eps=1.0, V=-2 / 3, W=0.0)          # chi = |V|(N-1)/eps = 2
    H = lipkin_level_encoding(lip)
    w, V = np.linalg.eigh(H.matrix())
    Eq = np.linalg.eigvalsh(lip.quasispin_matrix())
    assert abs(w[0] - Eq[0]) < 1e-10 and abs(w[0] + 3.055050) < 1e-6
    alpha = np.arccos(1 / 2)                                  # cos(alpha) = 1/chi, Table 4.lipkinmf
    ry = G.Ry(alpha)
    mean_field = StateVector.product(*(4 * [ry @ np.array([1, 0])])).psi
    ref = StateVector.from_bits("0000").psi
    E_mf = np.real(np.vdot(mean_field, H.matrix() @ mean_field))
    print(f"  mean-field energy {E_mf:.6f} (Table 4.lipkinmf: -2.500000), exact {w[0]:.6f}, J = 2 eigenvalues {Eq}")
    assert abs(E_mf + 2.5) < 1e-6
    states = [("reference $|0000\\rangle$", ref), (r"mean field, $\alpha=\pi/3$", mean_field), ("exact ground state", V[:, 0])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    spectro = []
    for ax, (label, st) in zip(axes, states):
        q = run_qpe(H, st, 8)
        p = q.distribution()
        c = np.abs(V.conj().T @ st) ** 2
        ax.bar(q.E, p, width=0.8 * (q.E[1] - q.E[0]))
        for e in Eq:
            ax.axvline(e, color="gray", ls=":", lw=0.7)
        ax.set_xlim(-3.5, 2.0); ax.set_xlabel("$E$"); ax.set_title(label, fontsize=9)
        # peaks: the weight within half a grid step of every J=2 eigenvalue
        weights = []
        for e in Eq:
            k = int(np.argmin(np.abs(q.E - e)))
            weights.append(p[max(k - 1, 0):k + 2].sum())
        overlaps = [c[np.argmin(np.abs(w - e))] if np.min(np.abs(w - e)) < 1e-8 else 0 for e in Eq]
        # degenerate levels: sum the overlaps of all eigenvectors at that energy
        overlaps = [c[np.abs(w - e) < 1e-8].sum() for e in Eq]
        spectro.append((label, overlaps, weights))
        k0 = int(np.argmin(np.abs(q.E - w[0])))
        fid = abs(np.vdot(V[:, 0], q.collapse(k0))) ** 2
        print(f"  {label:>28s}: |c_j|^2 over the J=2 levels {np.round(overlaps, 4)}; histogram weights {np.round(weights, 4)}; "
              f"fidelity of the collapsed register with the ground state after reading the E_0 bits: {fid:.4f}")
        assert fid > 0.99
    axes[0].set_ylabel("probability")
    savefig(fig, 6, "spectroscopy")
    print("\nmodel phase-estimation checks passed")
