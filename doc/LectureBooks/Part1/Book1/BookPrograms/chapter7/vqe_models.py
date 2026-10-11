"""Chapter 7, Secs. 7.6-7.7: VQE on the Hamiltonians of chapters 4 and 6.

* the two small Hamiltonians: the tapered one-qubit pairing Hamiltonian of
  Eq. (4.tapered) with the R_y ansatz (exact by construction) and the
  two-mode Hamiltonian of Eq. (4.twoqubitH) with the hardware-efficient
  ansatz, which finds the vacuum, and with the number-conserving single
  excitation of Sec. 4.5.5, which finds the one-particle ground state --
  Table 7.small;
* the four models of Table 6.models with the hardware-efficient ansatz of
  increasing depth (best of four restarts), against the lowest eigenvalue of
  the register and the energy of the physical sector -- Table 7.hea;
* the same models with ansätze that know the physics: the Hamiltonian-
  variational ansatz for the Lipkin, Heisenberg and Hubbard models and the
  pair-excitation circuit of chapter 4 for the pairing model, with the
  overlap of the optimised state with the exact ground state, which is what
  a subsequent phase-estimation run would see -- Table 7.physics,
  Fig. 7.convergence;
* the comparison with the phase-estimation energies of Table 6.qpemodels.

About one minute (the eight-qubit Hubbard runs dominate).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from pauli import PauliSum
from fockspace import FockSpace
from models import LipkinModel, PairingModel, HubbardModel, HeisenbergModel
from mapped_models import lipkin_level_encoding
from pauli_circuits import compact_pairing
from qpe_models import book_models, hubbard_ring_spin_major, sector_ground_state, split_identity
from vqe import VQE, HardwareEfficient, ProductState, PauliRotationAnsatz, run_restarts
from bookplot import savefig, plt


# ------------------------------------------------------------ generators
def pair_generator(n, i, a):
    """exp(-i theta G) = exp(theta (S+_a S-_i - h.c.)), the pair-excitation gate of Eq. (4.pairgadget)."""
    return (-0.5) * (PauliSum.single(n, i, "Y") * PauliSum.single(n, a, "X")
                     - PauliSum.single(n, i, "X") * PauliSum.single(n, a, "Y"))


def single_generator(n, p, q):
    """exp(-i theta G) = exp(theta (a+_q a_p - h.c.)) in the Jordan-Wigner encoding, Eq. (4.singlegadget)."""
    jw_yx = ["I"] * n; jw_xy = ["I"] * n
    for k in range(p + 1, q):
        jw_yx[k] = jw_xy[k] = "Z"
    jw_yx[p], jw_yx[q] = "Y", "X"
    jw_xy[p], jw_xy[q] = "X", "Y"
    return (-0.5) * (PauliSum({"".join(jw_yx): 1.0}) - PauliSum({"".join(jw_xy): 1.0}))


def hva_generators(H, layers):
    """The Hamiltonian-variational ansatz: the traceless Hamiltonian split into
    its groups of mutually commuting strings, one angle per group per layer."""
    hI, Ht = split_identity(H)
    groups = Ht.commuting_groups(qubitwise=False)
    gens = [PauliSum({k: Ht.terms[k] for k in g}, H.n) for g in groups]
    return [g for _ in range(layers) for g in gens], len(groups)


def overlap_with_ground(psi, H, sector=None):
    """|<E_0|psi>|^2 summed over the (possibly degenerate) ground space of H or of its sector."""
    if sector is None:
        w, V = np.linalg.eigh(H.matrix())
        space = V[:, np.abs(w - w[0]) < 1e-8]
        return float(np.sum(np.abs(space.conj().T @ psi) ** 2))
    idx = FockSpace(H.n).sector(sector)
    w, V = np.linalg.eigh(H.matrix()[np.ix_(idx, idx)])
    space = V[:, np.abs(w - w[0]) < 1e-8]
    return float(np.sum(np.abs(space.conj().T @ psi[idx]) ** 2))


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # ============================================ the small Hamiltonians (Table 7.small)
    print("=== The tapered one-qubit pairing Hamiltonian H = I/2 + Z - X/2 with the R_y ansatz")
    H1 = PauliSum({"I": 0.5, "Z": 1.0, "X": -0.5})
    E1 = 0.5 - np.sqrt(5) / 2
    ans1 = ProductState(1)
    theta, E, vqe = run_restarts(H1, ans1, restarts=3)
    print(f"  VQE {E:.10f}, exact {E1:.10f}, error {E - E1:.1e}, angle {theta[0]:.6f} (tan(theta) = {np.tan(theta[0]):.4f}, expected -1/2 up to the branch)")
    assert abs(E - E1) < 1e-9

    print("\n=== The two-mode Hamiltonian of Eq. (4.twoqubitH): hardware-efficient ansatz against the single excitation")
    H2 = PauliSum({"II": 1.625, "ZI": -0.625, "IZ": -1.125, "ZZ": 0.125, "XX": 0.15, "YY": 0.15})
    w2 = np.linalg.eigvalsh(H2.matrix())
    E0_sector, _, ws = sector_ground_state(H2, 1)
    rows = []
    for layers in (1, 2):
        hea = HardwareEfficient(2, layers)
        theta, E, vqe = run_restarts(H2, hea, restarts=4)
        psi = hea.state(theta)
        n_particles = float(np.real(np.vdot(psi, (PauliSum.identity(2) - 0.5 * (PauliSum({"ZI": 1.0}) + PauliSum({"IZ": 1.0}))).matrix() @ psi)))
        rows.append((f"hardware-efficient, {layers} layer{'s' if layers > 1 else ''}", hea.n_params, E, n_particles))
    single = PauliRotationAnsatz(2, "10", [single_generator(2, 0, 1)])
    theta, E, vqe = run_restarts(H2, single, restarts=3, scale=1.0)
    rows.append(("single excitation from |10>", 1, E, 1.0))
    print(f"{'ansatz':>32s} {'params':>6s} {'E_VQE':>10s} {'<N>':>5s}")
    for name, p, E, N in rows:
        print(f"{name:>32s} {p:6d} {E:10.6f} {N:5.2f}")
    print(f"  spectrum of the register {w2}; one-particle ground state {E0_sector:.6f}; the single excitation gives theta = {theta[0]:.4f}")
    assert abs(rows[0][2] - w2[0]) < 1e-8 and abs(rows[-1][2] - E0_sector) < 1e-8

    # ============================================ the four models with the hardware-efficient ansatz (Table 7.hea)
    print("\n=== The four models with the hardware-efficient ansatz, best of four restarts (Table 7.hea)")
    models = book_models()
    register_min = {name: np.linalg.eigvalsh(H.matrix())[0] for name, H, E0, psi, sector in models}
    hea_rows = {}
    print(f"{'model':>16s} {'layers':>6s} {'params':>6s} {'CNOTs':>5s} {'E_VQE':>11s} {'register min':>12s} {'sector E0':>10s} {'|<E0|psi>|^2':>12s}")
    for name, H, E0, psi0, sector in models:
        depths = (1, 2, 3) if H.n <= 4 else (1, 2)
        for layers in depths:
            hea = HardwareEfficient(H.n, layers)
            theta, E, vqe = run_restarts(H, hea, restarts=4 if H.n <= 4 else 2, maxiter=300)
            ov = overlap_with_ground(hea.state(theta), H, sector)
            hea_rows[(name, layers)] = (hea.n_params, hea.cnots(), E, ov)
            print(f"{name:>16s} {layers:6d} {hea.n_params:6d} {hea.cnots():5d} {E:11.6f} {register_min[name]:12.6f} {E0:10.6f} {ov:12.4f}")
            assert E >= register_min[name] - 1e-9

    # ============================================ ansätze that know the physics (Table 7.physics)
    print("\n=== Ansätze that know the physics (Table 7.physics)")
    physics = {}
    histories = {}
    print(f"{'model':>16s} {'ansatz':>28s} {'params':>6s} {'E_VQE':>12s} {'exact E0':>12s} {'error':>8s} {'|<E0|psi>|^2':>12s}")
    for name, H, E0, psi0, sector in models:
        rows = []
        if name.startswith("Lipkin"):
            rows.append(("mean field, one R_y angle", ProductState(4, shared=True), None, 0.5))
            for L in (1, 2, 3):
                gens, ng = hva_generators(H, L)
                rows.append((f"Hamiltonian-variational, {L} layer{'s' if L > 1 else ''}", PauliRotationAnsatz(4, "0000", gens), None, 0.3))
        elif name.startswith("pairing"):
            pairs = [(i, a) for i in (0, 1) for a in (2, 3)]
            gens = [pair_generator(4, i, a) for i, a in pairs]
            rows.append(("four pair excitations (UCCD)", PauliRotationAnsatz(4, "1100", gens), 2, 1.0))
            all_pairs = [pair_generator(4, i, a) for i in range(4) for a in range(i + 1, 4)]
            rows.append(("all six pair transfers", PauliRotationAnsatz(4, "1100", all_pairs), 2, 1.0))
        elif name.startswith("Hubbard"):
            hub, _ = hubbard_ring_spin_major()
            hI, Ht = split_identity(H)
            hop = PauliSum({k: c for k, c in Ht.terms.items() if "X" in k or "Y" in k}, 8)
            onsite = PauliSum({k: c for k, c in Ht.terms.items() if "X" not in k and "Y" not in k}, 8)
            # reference: the weak-coupling ground state of the half-filled ring (U = 0.01 t), the
            # member of the six-fold degenerate free ground manifold that the interaction selects
            _, H0 = hubbard_ring_spin_major(U=0.01)
            E00, psi_free, _ = sector_ground_state(H0, 4)
            print(f"  Hubbard reference: the U = 0.01 ground state, overlap with the U = 4 ground state {overlap_with_ground(psi_free, H, 4):.4f}")
            for L in (1, 2, 3):
                rows.append((f"Hamiltonian-variational, {L} layer{'s' if L > 1 else ''}",
                             PauliRotationAnsatz(8, psi_free, [g for _ in range(L) for g in (onsite, hop)]), 4, 0.3))
        else:
            # Heisenberg: start from the singlet covering of the even bonds and alternate the two bond groups
            singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
            ref = np.kron(singlet, singlet)
            gens, ng = hva_generators(H, 1)
            odd = [g for g in gens if "IXXI" in g.terms][0]; even = [g for g in gens if "XXII" in g.terms][0]
            for L in (1, 2):
                rows.append((f"Hamiltonian-variational, {L} layer{'s' if L > 1 else ''}",
                             PauliRotationAnsatz(4, ref, [g for _ in range(L) for g in (odd, even)]), None, 0.3))
        for label, ans, sec, scale in rows:
            theta, E, vqe = run_restarts(H, ans, restarts=4 if H.n <= 4 else 2, scale=scale, maxiter=400)
            ov = overlap_with_ground(ans.state(theta), H, sec)
            physics[(name, label)] = (ans.n_params, E, E0, ov)
            print(f"{name:>16s} {label:>28s} {ans.n_params:6d} {E:12.8f} {E0:12.8f} {E - E0:8.1e} {ov:12.6f}")
            assert E >= E0 - 1e-8
        # convergence history of the last (deepest) physics ansatz, gradient descent from the same start
        # (the Hubbard generators are sums of many strings with |h| up to 4, so the step is ten times smaller)
        v = VQE(H, ans)
        v.gradient_descent(ans.random_parameters(np.random.default_rng(2), scale), lr=0.01 if H.n == 8 else 0.1, iters=120)
        histories[name] = (np.array(v.history), E0)
    # the pairing circuit reproduces the chapter-4 UCCD energy
    assert abs(physics[("pairing, $L=4$", "four pair excitations (UCCD)")][1] - 0.63698698) < 5e-6
    assert abs(physics[("pairing, $L=4$", "all six pair transfers")][1] - 0.63554847) < 1e-6
    assert abs(physics[("Hubbard ring", "Hamiltonian-variational, 2 layers")][1] + 2.10274848) < 1e-6
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, (hist, E0) in histories.items():
        ax.semilogy(np.arange(len(hist)), np.maximum(hist - E0, 1e-12), label=name.replace("$", ""))
    ax.set_xlabel("gradient-descent iteration"); ax.set_ylabel(r"$E(\theta) - E_0$"); ax.legend(fontsize=8)
    savefig(fig, 7, "convergence")

    # ============================================ against phase estimation (Table 7.qpevsvqe)
    print("\n=== The four models: phase estimation (m = 10, Table 6.qpemodels) against the best variational energies")
    qpe_err = {"Lipkin, $N=4$": 5.9e-3, "pairing, $L=4$": 7.3e-3, "Hubbard ring": 1.7e-2, "Heisenberg ring": 1.3e-3}
    print(f"{'model':>16s} {'QPE error':>10s} {'HEA error':>10s} {'physics error':>14s} {'physics overlap':>15s}")
    for name, H, E0, psi0, sector in models:
        hea_best = min(v[2] for (nm, L), v in hea_rows.items() if nm == name)
        phys = [v for (nm, lab), v in physics.items() if nm == name]
        best = min(phys, key=lambda v: v[1])
        print(f"{name:>16s} {qpe_err[name]:10.1e} {hea_best - E0:+10.1e} {best[1] - E0:14.1e} {best[3]:15.6f}")
    print("\nmodel VQE checks passed")
