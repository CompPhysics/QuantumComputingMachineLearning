"""Chapter 4, Secs. 4.5-4.6: from Pauli strings to circuits and measurements.

* the circuit for exp(-i theta P) for any Pauli string P (basis change,
  CNOT ladder, one Rz, Fig. 4.gadget), checked against the matrix exponential;
* one Trotter step of the compact pairing Hamiltonian and of the Heisenberg
  ring as gate sequences, with gate counts (Table 4.trottergates) and the
  splitting error against the step length;
* preparation of the reference determinant and estimation of the energy from
  shots, one measurement setting per group of qubit-wise commuting strings
  (Table 4.shots);
* the pair-excitation circuit exp(theta (S^+_a S^-_i - h.c.)) and a
  Trotterised unitary coupled-cluster circuit for the pairing model, whose
  optimised energy reproduces the UCCD numbers of chapter 3 (Table 4.uccd).
Runs in about ten seconds (the repeated shot experiments dominate).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from pauli import PauliSum, string_matrix
from circuit import Circuit
from statevector import StateVector
from matrixfunctions import expm_hermitian, expm_antihermitian
from models import PairingModel, HeisenbergModel
from optimize import minimize
from bookplot import savefig, plt


# ------------------------------------------------------------ Pauli gadget
def pauli_rotation(circ, label, theta):
    """Append gates implementing exp(-i theta P) for the Pauli string `label`, Fig. 4.gadget:
    rotate every X (Y) qubit into the Z basis with H (S^dagger H), collect the
    parity on the last qubit with a CNOT ladder, apply Rz(2 theta), undo."""
    support = [q for q, c in enumerate(label) if c != "I"]
    if not support:
        return circ                                   # a global phase, no gate
    for q in support:
        if label[q] == "X":
            circ.h(q)
        elif label[q] == "Y":
            circ.add(G_Sdg(q)); circ.h(q)
    for a, b in zip(support[:-1], support[1:]):
        circ.cnot(a, b)
    circ.rz(2 * theta, support[-1])
    for a, b in reversed(list(zip(support[:-1], support[1:]))):
        circ.cnot(a, b)
    for q in reversed(support):
        if label[q] == "X":
            circ.h(q)
        elif label[q] == "Y":
            circ.h(q); circ.add(G_S(q))
    return circ


def G_S(q):
    from circuit import Gate
    return Gate("S", G.S, q)


def G_Sdg(q):
    from circuit import Gate
    return Gate("S^", G.S.conj().T, q)


def trotter_step(H, dt, order=1):
    """One Lie-Trotter (order 1) or Strang (order 2) step of exp(-i H dt), string by string."""
    circ = Circuit(H.n)
    labels = [k for k in H.terms if any(c != "I" for c in k)]
    if order == 1:
        for k in labels:
            pauli_rotation(circ, k, H.terms[k].real * dt)
    else:
        for k in labels[:-1]:
            pauli_rotation(circ, k, H.terms[k].real * dt / 2)
        pauli_rotation(circ, labels[-1], H.terms[labels[-1]].real * dt)
        for k in reversed(labels[:-1]):
            pauli_rotation(circ, k, H.terms[k].real * dt / 2)
    return circ


def gate_counts(circ):
    two = sum(1 for g in circ.gates if len(g.qubits) == 2)
    return len(circ.gates), two, circ.depth


def prepare_reference(n, occupied):
    circ = Circuit(n)
    for q in occupied:
        circ.x(q)
    return circ


# ------------------------------------------------------------- measurement
def measure_group(state, group, shots, rng):
    """Estimate <P> for every string P of a qubit-wise commuting `group` from
    `shots` samples of one circuit: rotate each qubit into the basis the group
    needs, sample bit strings, and average (-1)^(parity on the support of P)."""
    n = state.n
    basis = ["I"] * n
    for k in group:
        for q, c in enumerate(k):
            if c != "I":
                basis[q] = c
    rotated = state.copy()
    for q, c in enumerate(basis):
        if c == "X":
            rotated.h(q)
        elif c == "Y":
            rotated.apply(G.S.conj().T, q); rotated.h(q)
    counts = rotated.sample(shots, rng)
    estimates = {}
    for k in group:
        support = [q for q, c in enumerate(k) if c != "I"]
        total = 0
        for bits, m in counts.items():
            parity = sum(int(bits[q]) for q in support) % 2
            total += m * (1 - 2 * parity)
        estimates[k] = total / shots
    return estimates


def energy_from_shots(H, state, shots_per_group, rng):
    groups = H.commuting_groups(qubitwise=True)
    E = 0.0
    for g in groups:
        est = measure_group(state, g, shots_per_group, rng)
        for k in g:
            E += H.terms[k].real * est[k]
    return E, len(groups)


# ---------------------------------------------------- pair-excitation gadget
def pair_excitation(circ, i, a, theta):
    """exp(theta (S^+_a S^-_i - h.c.)) in the one-qubit-per-level encoding:
    S^+_a S^-_i - h.c. = (i/2)(Y_i X_a - X_i Y_a), two commuting rotations, Eq. (4.pairgadget)."""
    n = circ.n
    yx = ["I"] * n; yx[i] = "Y"; yx[a] = "X"
    xy = ["I"] * n; xy[i] = "X"; xy[a] = "Y"
    pauli_rotation(circ, "".join(yx), -theta / 2)
    pauli_rotation(circ, "".join(xy), +theta / 2)
    return circ


def single_excitation(circ, p, q, theta):
    """exp(theta (a+_q a_p - h.c.)) for modes p < q in the Jordan-Wigner encoding:
    a+_q a_p - h.c. = (i/2)(Y_p Z...Z X_q - X_p Z...Z Y_q), two commuting rotations, Eq. (4.singlegadget)."""
    n = circ.n
    xy = ["I"] * n; yx = ["I"] * n
    for k in range(p + 1, q):
        xy[k] = yx[k] = "Z"
    xy[p], xy[q] = "X", "Y"
    yx[p], yx[q] = "Y", "X"
    pauli_rotation(circ, "".join(yx), -theta / 2)
    pauli_rotation(circ, "".join(xy), +theta / 2)
    return circ


def compact_pairing(pm):
    """Eq. (4.jwpairingcompact) as a PauliSum on L qubits."""
    L = pm.L
    H = PauliSum(n_qubits=L)
    for p in range(L):
        H = H + (pm.xi * p - pm.g / 4) * (PauliSum.identity(L) - PauliSum.single(L, p, "Z"))
        for q in range(p + 1, L):
            H = H - pm.g / 4 * (PauliSum.single(L, p, "X") * PauliSum.single(L, q, "X")
                                + PauliSum.single(L, p, "Y") * PauliSum.single(L, q, "Y"))
    return H.simplify()


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # ------------------------------------------------- exp(-i theta P) circuits
    print("=== Circuits for exp(-i theta P) (Fig. 4.gadget)")
    for label in ("ZZ", "XX", "YY", "XZZY", "XYXY"):
        theta = 0.37
        circ = pauli_rotation(Circuit(len(label)), label, theta)
        U = circ.unitary()
        exact = expm_hermitian(string_matrix(label), -1j * theta)
        assert np.allclose(U, exact)
        print(f"  {label}: {len(circ.gates)} gates, {sum(1 for g in circ.gates if len(g.qubits) == 2)} CNOTs, depth {circ.depth}")
    circ = pauli_rotation(Circuit(2), "YY", 0.37)
    print(circ.draw())

    # ------------------------------------------------------ Trotter steps
    print("\n=== One Trotter step as a circuit (Table 4.trottergates)")
    pm = PairingModel(2, g=1.0)
    Hc = compact_pairing(pm)
    heis = HeisenbergModel(4)
    Hh = PauliSum.from_matrix(heis.full_matrix())
    print(f"{'Hamiltonian':>26s} {'strings':>8s} {'gates':>6s} {'CNOTs':>6s} {'depth':>6s}")
    for name, H in (("pairing L=2, one per level", Hc), ("Heisenberg ring, 4 sites", Hh)):
        circ = trotter_step(H, 0.1)
        ng, n2, d = gate_counts(circ)
        print(f"{name:>26s} {H.n_strings():8d} {ng:6d} {n2:6d} {d:6d}")
    print("pairing model, L = 2, one qubit per level, one first-order step:")
    print(trotter_step(Hc, 0.1).draw())
    t = 1.0
    print("\n  splitting error of n steps for the pairing Hamiltonian, t = 1:")
    print(f"{'n':>4s} {'first order':>12s} {'second order':>13s}")
    Hc0 = Hc - Hc.terms["II"]                      # the identity string is a global phase
    exact = expm_hermitian(Hc0.matrix(), -1j * t)
    err = {1: [], 2: []}
    ns = (1, 2, 4, 8, 16)
    for n in ns:
        for order in (1, 2):
            U = np.linalg.matrix_power(trotter_step(Hc, t / n, order).unitary(), n)
            err[order].append(np.linalg.norm(U - exact, 2))
        print(f"{n:4d} {err[1][-1]:12.3e} {err[2][-1]:13.3e}")
    assert err[1][-1] / err[1][-2] < 0.6 and err[2][-1] / err[2][-2] < 0.3

    # ------------------------------------------- a first time evolution
    print("\n=== Trotter circuit dynamics: pair transfer in the two-level model (Fig. 4.dynamics)")
    ref = prepare_reference(2, [0]).run()                 # |10>: the pair in level 1
    times = np.linspace(0, 8, 161)
    Hm = Hc0.matrix()
    exact_p = [abs(np.vdot(StateVector.from_bits("01").psi, expm_hermitian(Hm, -1j * tt) @ ref.psi)) ** 2 for tt in times]
    dyn = {}
    for dt, order in ((0.5, 1), (0.5, 2), (0.1, 1)):
        step = trotter_step(Hc, dt, order)
        state = ref.copy(); probs = [0.0]; ts = [0.0]
        for k in range(int(8 / dt)):
            state = step.run(state)
            ts.append((k + 1) * dt); probs.append(state.probabilities()[int("01", 2)])
        dyn[(dt, order)] = (ts, probs)
        worst = max(abs(pr - abs(np.vdot(StateVector.from_bits("01").psi, expm_hermitian(Hm, -1j * tt) @ ref.psi)) ** 2)
                  for tt, pr in zip(ts, probs))
        print(f"  dt = {dt}, order {order}: {int(8 / dt)} steps, {gate_counts(step)[1] * int(8 / dt)} CNOTs, "
              f"largest error in P(t) = {worst:.3e}")
    print(f"  exact: P(t) oscillates with period {2 * np.pi / np.sqrt(5):.4f} = 2 pi / sqrt(5)")

    print("  Heisenberg ring, 4 sites, from the Neel state |0101>: staggered magnetisation")
    heisH = Hh - Hh.terms.get("IIII", 0.0)
    neel = StateVector.from_bits("0101")
    stag = PauliSum({"ZIII": 0.25, "IZII": -0.25, "IIZI": 0.25, "IIIZ": -0.25})
    hd = {}
    for dt, order in ((0.25, 1), (0.25, 2)):
        step = trotter_step(heisH, dt, order)
        state = neel.copy(); ms = [stag.expectation(state.psi).real]
        for k in range(int(6 / dt)):
            state = step.run(state); ms.append(stag.expectation(state.psi).real)
        hd[(dt, order)] = (np.arange(len(ms)) * dt, ms)
    ht = np.linspace(0, 6, 121)
    hexact = [stag.expectation(expm_hermitian(heisH.matrix(), -1j * tt) @ neel.psi).real for tt in ht]
    worst = max(abs(m - stag.expectation(expm_hermitian(heisH.matrix(), -1j * tt) @ neel.psi).real)
                for tt, m in zip(*hd[(0.25, 2)]))
    print(f"  second order, dt = 0.25: {gate_counts(trotter_step(heisH, 0.25, 2))[1]} CNOTs per step, largest error {worst:.2e}")

    # --------------------------------------------- reference state and shots
    print("\n=== Energy from shots, pairing L = 2, one pair (Table 4.shots)")
    ref = prepare_reference(2, [0]).run()          # |10>: level 1 filled with one pair
    E_ref = Hc.expectation(ref.psi).real
    _, H2 = pm.seniority_zero_matrix(1)
    assert abs(E_ref - H2[0, 0]) < 1e-12
    groups = Hc.commuting_groups(qubitwise=True)
    print(f"  qubit-wise commuting groups of the {Hc.n_strings()} strings: {groups}")
    # the exact ground state, to have something with fluctuations to measure
    E, V = np.linalg.eigh(Hc.matrix())
    psi = StateVector(2, V[:, 0])
    # the ground state lies in the one-pair sector; use the exact energy from the 2x2 block
    E0 = np.linalg.eigvalsh(H2)[0]
    assert abs(E[0] - E0) < 1e-12
    print(f"{'shots/group':>12s} {'one estimate':>13s} {'rms error (20 runs)':>20s} {'rms x sqrt(shots)':>18s}")
    for shots in (100, 1000, 10000, 100000):
        ests = np.array([energy_from_shots(Hc, psi, shots, rng)[0] for _ in range(20)])
        rms = np.sqrt(np.mean((ests - E0) ** 2))
        print(f"{shots:12d} {ests[0]:13.5f} {rms:20.2e} {np.sqrt(shots) * rms:18.3f}")
    ngroups = len(groups)
    print(f"  exact energy {E0:.8f}; {ngroups} measurement settings; the error falls as 1/sqrt(shots)")

    # ------------------------------------------- variances and shot allocation
    print("\n=== Variances of the strings and the shots needed for 1e-3 (Table 4.variances)")
    pm4 = PairingModel(4, g=1.0)
    H4 = compact_pairing(pm4)
    E4, V4 = np.linalg.eigh(H4.matrix())
    psi4 = V4[:, 0]
    print(f"{'string':>8s} {'h':>8s} {'<P>':>8s} {'Var P':>8s}")
    var_total = 0.0; naive = 0.0
    for k, c in sorted(H4.terms.items(), key=lambda kv: -abs(kv[1])):
        P = PauliSum({k: 1.0}); m = P.expectation(psi4).real; var = 1 - m ** 2
        var_total += abs(c) * np.sqrt(var); naive += abs(c)
        if k in ("IIII", "ZIII", "IIIZ", "XXII", "XIIX", "YYII"):
            print(f"{k:>8s} {c:8.4f} {m:8.4f} {var:8.4f}")
    eps = 1e-3
    print(f"  {H4.n_strings()} strings; optimal allocation S_a ~ |h_a| sigma_a: total shots {var_total ** 2 / eps ** 2:.2e};"
          f" naive bound (sum|h|)^2/eps^2 = {naive ** 2 / eps ** 2:.2e}")

    # -------------------------------------------------- pair-excitation gate
    print("\n=== The pair-excitation circuit exp(theta (S+_1 S-_0 - h.c.)) (Fig. 4.pairgadget)")
    theta = 0.6
    circ = pair_excitation(Circuit(2), 0, 1, theta)
    sp = lambda q: PauliSum.single(2, q, "X", 0.5) + PauliSum.single(2, q, "Y", -0.5j)
    T = sp(1) * sp(0).dagger()
    kappa = (T - T.dagger()).simplify()
    print("  generator S+_1 S-_0 - h.c. =", {k: complex(c) for k, c in kappa.terms.items()})
    assert np.allclose(circ.unitary(), expm_antihermitian(theta * kappa.matrix()))
    print(circ.draw())
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 181)
    Es = []
    for th in thetas:
        state = pair_excitation(prepare_reference(2, [0]), 0, 1, th).run()
        Es.append(Hc.expectation(state.psi).real)
    Es = np.array(Es)
    # |Psi(theta)> = cos(theta)|10> + sin(theta)|01>, Eq. (4.pairstate)
    assert np.allclose(Es, H2[0, 0] * np.cos(thetas) ** 2 + H2[1, 1] * np.sin(thetas) ** 2 + H2[0, 1] * np.sin(2 * thetas))
    k = np.argmin(Es)
    print(f"  E(theta) minimal at theta = {thetas[k]:.3f}: {Es[k]:.6f}; exact {E0:.6f}; "
          f"E(theta) = {H2[0,0]:.3f} cos^2 + {H2[1,1]:.3f} sin^2 + {H2[0,1]:.3f} sin 2theta")

    # ------------------------------------------- fermionic single excitations
    print("\n=== The single-excitation gate exp(theta (a+_q a_p - h.c.)) (Fig. 4.singlegadget)")
    from jordanwigner import JordanWigner
    jw = JordanWigner(4)
    for (p, q) in ((0, 1), (0, 3)):
        kappa = (jw.create(q) * jw.annihilate(p) - jw.create(p) * jw.annihilate(q)).simplify()
        circ = single_excitation(Circuit(4), p, q, 0.45)
        assert np.allclose(circ.unitary(), expm_antihermitian(0.45 * kappa.matrix()))
        print(f"  modes {p},{q}: generator {dict((k, complex(c)) for k, c in kappa.terms.items())}, "
              f"{gate_counts(circ)[1]} CNOTs")
    # the two-mode Hamiltonian: one rotation from |10> reaches the one-particle ground state
    jw2 = JordanWigner(2)
    e0, e1, U, t = 1.0, 2.0, 0.5, 0.3
    H2m = (e0 * jw2.number(0) + e1 * jw2.number(1) + U * (jw2.number(0) * jw2.number(1))
           + t * (jw2.create(0) * jw2.annihilate(1) + jw2.create(1) * jw2.annihilate(0))).simplify()
    energy = lambda th: H2m.expectation(single_excitation(prepare_reference(2, [0]), 0, 1, th[0]).run().psi).real
    th, E2m, _ = minimize(energy, np.array([0.1]))
    print(f"  two-mode Hamiltonian: E(theta) minimal at theta = {th[0]:.4f}, E = {E2m:.6f} "
          f"(exact one-particle ground state {(e0 + e1) / 2 - np.hypot((e0 - e1) / 2, t):.6f})")
    assert abs(E2m - ((e0 + e1) / 2 - np.hypot((e0 - e1) / 2, t))) < 1e-8

    # ------------------------------------------- the Lipkin mean field as a product state
    print("\n=== Lipkin mean field as a product state on four qubits (Table 4.lipkinmf)")
    from mapped_models import lipkin_level_encoding
    from models import LipkinModel
    from hartreefock import HartreeFock
    N, eps = 4, 1.0
    print(f"{'chi':>5s} {'alpha':>8s} {'E(product state)':>17s} {'E_HF (chapter 3)':>17s} {'E_exact':>10s}")
    for chi in (0.5, 1.0, 1.5, 2.0, 2.5):
        lip = LipkinModel(N, eps=eps, V=-chi * eps / (N - 1), W=0.0)
        Hlev = lipkin_level_encoding(lip)
        def product_energy(a):
            c = Circuit(N)
            for q in range(N):
                c.ry(a[0], q)
            return Hlev.expectation(c.run().psi).real
        best = min((minimize(product_energy, np.array([a0]))[:2] for a0 in (0.0, 0.8, 1.6)), key=lambda r: r[1])
        hf = HartreeFock(lip.h, lip.v, N)
        E_hf = min(hf.solve(np.eye(lip.n_modes)[:, np.argsort(np.diag(lip.h), kind="stable")])[2],
                   hf.solve(np.linalg.qr(np.random.default_rng(3).normal(size=(8, 8)))[0])[2])
        E_ex = np.linalg.eigvalsh(lip.quasispin_matrix())[0]
        print(f"{chi:5.2f} {abs(best[0][0]):8.4f} {best[1]:17.6f} {E_hf:17.6f} {E_ex:10.6f}")
        assert abs(best[1] - E_hf) < 1e-5

    # --------------------------------------------- UCC circuit, L = 4, two pairs
    print("\n=== Trotterised UCC circuit for the pairing model, L = 4, two pairs (Table 4.uccd)")
    print(f"{'g':>5s} {'E_ref':>9s} {'circuit UCCD':>13s} {'chapter 3':>11s} {'FCI':>12s} {'CNOTs':>6s}")
    chapter3 = {0.25: 1.73046012, 0.5: 1.41679562, 1.0: 0.63698698, 1.5: -0.33675149, 2.0: -1.45347633}
    pairs = [(i, a) for i in (0, 1) for a in (2, 3)]

    def ucc_circuit(t):
        circ = prepare_reference(4, [0, 1])
        for (i, a), tk in zip(pairs, t):
            pair_excitation(circ, i, a, tk)
        return circ

    ucc_rows = []
    for g in (0.25, 0.5, 1.0, 1.5, 2.0):
        pm4 = PairingModel(4, g=g)
        H4 = compact_pairing(pm4)
        Hm = H4.matrix()
        energy = lambda t: float(np.real(np.vdot(ucc_circuit(t).run().psi, Hm @ ucc_circuit(t).run().psi)))
        t, E_ucc, it = minimize(energy, np.zeros(4), tol=1e-12, gtol=1e-8)
        _, Hs = pm4.seniority_zero_matrix(2)
        E_fci = np.linalg.eigvalsh(Hs)[0]
        ncx = sum(1 for gg in ucc_circuit(t).gates if len(gg.qubits) == 2)
        print(f"{g:5.2f} {Hs[0,0]:9.4f} {E_ucc:13.8f} {chapter3[g]:11.8f} {E_fci:12.8f} {ncx:6d}")
        assert abs(E_ucc - chapter3[g]) < 2e-6
        ucc_rows.append((g, E_ucc, E_fci))
    print("  the circuit is the n = 1 Trotter product of chapter 3 with re-optimised amplitudes;")
    ng, n2, d = gate_counts(ucc_circuit(t))
    print(f"  {ng} gates, {n2} CNOTs, depth {d}, four parameters")

    # ------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(times, exact_p, "k-", label="exact")
    for (dt, order), (ts, probs) in dyn.items():
        ax.plot(ts, probs, "o" if order == 1 else "s", ms=3 if dt < 0.2 else 4, ls="--", lw=0.8,
                label=f"Trotter, $\\Delta t={dt}$, order {order}")
    ax.set_xlabel("$t$"); ax.set_ylabel("$P(t) = |\\langle 01|\\Psi(t)\\rangle|^2$"); ax.legend(fontsize=7)
    ax.set_title("pair transfer, two-level pairing model, $g=1$", fontsize=9)
    ax2 = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
    ax2.plot(ht, hexact, "k-", lw=1)
    for (dt, order), (ts, ms) in hd.items():
        ax2.plot(ts, ms, "o" if order == 1 else "s", ms=2.5, ls="none")
    ax2.set_title("Heisenberg ring from $|0101\\rangle$", fontsize=7); ax2.tick_params(labelsize=6)
    ax2.set_ylabel("$m_s(t)$", fontsize=7)
    print("figure written to", savefig(fig, 4, "dynamics"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    ax.loglog(ns, err[1], "o-", label="first order"); ax.loglog(ns, err[2], "s-", label="second order")
    ax.set_xlabel("Trotter steps $n$"); ax.set_ylabel(r"$\|U_n - e^{-iHt}\|$")
    ax.set_title("pairing $L=2$, one qubit per level, $t=1$", fontsize=9); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(thetas, Es, label=r"$E(\theta)$"); ax.axhline(E0, color="k", ls="--", lw=1, label="exact")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("energy"); ax.legend(fontsize=8)
    ax.set_title("one pair-excitation gate, $L=2$, $g=1$", fontsize=9)
    print("figure written to", savefig(fig, 4, "circuits"))
