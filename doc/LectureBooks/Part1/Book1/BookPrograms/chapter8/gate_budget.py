"""Chapter 8, Sec. 8.7: circuits, gate counts and the depth budget.

Everything is compiled to single-qubit gates and CNOTs through the Pauli
rotation of Fig. 4.gadget: a string of weight w costs 2(w-1) CNOTs and one
R_z; controlling the rotation on an ancilla means controlling the R_z only,
two more CNOTs.

* the Jordan-Wigner strings of single and double excitations and the CNOT
  count of the UCCSD preparation (Table 8.uccsd, last column; Table 8.blocks);
* one Trotter step of the pairing Hamiltonian, Eq. (8.trottercost), plain and
  controlled, in the 2k-qubit Jordan-Wigner encoding and in the compact
  pair encoding of Eq. (4.jwpairingcompact);
* the projector term of the refinement path;
* circuit depth against CNOT count on the simulator of chapter 2;
* the budget of the pipeline against the UCCSD preparation (Table 8.depth)
  and the same for the three other models of Table 6.models;
* expected number of faults and success probability at a given two-qubit
  error rate (Table 8.faults).

About 1 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
import numpy as np
from pauli import PauliSum, weight, string_matrix
from jordanwigner import JordanWigner
from models import PairingModel
from circuit import Circuit
from pauli_circuits import pauli_rotation, compact_pairing, gate_counts
from matrixfunctions import expm_hermitian
from qpe_models import book_models, split_identity


def cnots(H, controlled=False, skip_identity=True):
    """CNOT count of one first-order step of exp(-i H dt): 2(w-1) per string, +2 if controlled."""
    total = 0
    for k in H.terms:
        w = weight(k)
        if w == 0:
            continue                        # a phase (a phase gate on the ancilla when controlled)
        total += 2 * (w - 1) + (2 if controlled else 0)
    return total


def excitation_strings(n, occ, virt):
    """The Pauli strings of tau - tau^dagger for tau = prod a_a^dagger prod a_i."""
    jw = JordanWigner(n)
    tau = PauliSum.identity(n)
    for a in virt:
        tau = tau * jw.create(a)
    for i in occ:
        tau = tau * jw.annihilate(i)
    return (tau - tau.dagger()).simplify()


def uccsd_pool(k, N):
    n = 2 * k
    occ, virt = range(N), range(N, n)
    singles = [((i,), (a,)) for i in occ for a in virt if i % 2 == a % 2]
    doubles = [((i, j), (a, b)) for i, j in itertools.combinations(occ, 2)
               for a, b in itertools.combinations(virt, 2) if i % 2 + j % 2 == a % 2 + b % 2]
    return singles + doubles


def uccsd_counts(k, N=4, pairs_only=False):
    """(operators, Pauli rotations, CNOTs) of the UCCSD state preparation."""
    ops = rot = cx = 0
    for occ, virt in uccsd_pool(k, N):
        if pairs_only and not (len(occ) == 2 and occ[0] % 2 == 0 and occ[1] == occ[0] + 1
                               and virt[0] % 2 == 0 and virt[1] == virt[0] + 1):
            continue
        A = excitation_strings(2 * k, occ, virt)
        ops += 1; rot += A.n_strings(); cx += cnots(A)
    return ops, rot, cx


def projector_empty(n, qubits):
    """prod_q (1 - n_q) = prod_q (I + Z_q)/2: the 'new levels empty' projector of the path."""
    P = PauliSum.identity(n)
    for q in qubits:
        P = P * ((PauliSum.identity(n) + PauliSum.single(n, q, "Z")) * 0.5)
    return P.simplify()


def second_order(hop_terms, hop_cost, diag_cost):
    """CNOTs of one symmetric step: 2J - 1 hop exponentials (the middle pair merges) and one
    diagonal layer (the half layers of neighbouring steps merge)."""
    return (2 * hop_terms - 1) * hop_cost + diag_cost


if __name__ == "__main__":
    # ---- the primitive
    label, phi = "XXXY", 0.731
    circ = pauli_rotation(Circuit(4), label, phi / 2)
    err = np.linalg.norm(circ.unitary() - expm_hermitian(string_matrix(label), -0.5j * phi), 2)
    ng, n2, depth = gate_counts(circ)
    print(f"exp(-i phi/2 {label}): {ng} gates, {n2} CNOTs, depth {depth}, error {err:.1e}")
    assert err < 1e-12 and n2 == 6

    # ---- excitation operators
    print("\nJordan-Wigner strings of tau - tau^dagger (k = 3, six qubits)")
    for occ, virt, name in (((0,), (4,), "single 0 -> 4"), ((1,), (5,), "single 1 -> 5"), ((2,), (4,), "single 2 -> 4"),
                            ((2, 3), (4, 5), "pair (2,3) -> (4,5)"), ((0, 1), (4, 5), "pair (0,1) -> (4,5)"),
                            ((0, 3), (4, 5), "double (0,3) -> (4,5)"), ((0, 2), (4, 5), "double (0,2) -> ... none")):
        if occ == (0, 2):
            continue
        A = excitation_strings(6, occ, virt)
        print(f"  {name:24s} {A.n_strings()} strings, weights {sorted(A.weights().items())}, {cnots(A)} CNOTs")
    print("\nUCCSD preparation, N = 4")
    print("  k   operators  rotations  CNOTs    pair doubles only: operators  CNOTs")
    for k in (3, 4):
        o, r, c = uccsd_counts(k)
        po, pr, pc = uccsd_counts(k, pairs_only=True)
        print(f"  {k}   {o:6d}   {r:8d}   {c:6d}                         {po:6d}   {pc:6d}")
        if k == 3: assert (o, r, c) == (8, 40, 272)
        if k == 4: assert (o, r, c) == (26, 160, 1312)

    # ---- one Trotter step of the pairing Hamiltonian
    print("\nOne first-order Trotter step of the pairing Hamiltonian: CNOTs plain / controlled")
    print("   k   Jordan-Wigner, 2k qubits      formula 24k(k-1)+2k     compact, k qubits")
    step = {}
    for k in (2, 3, 4):
        pm = PairingModel(k, g=1.0)
        Hjw = JordanWigner(2 * k).map_hamiltonian(pm.h, pm.v)
        Hc = compact_pairing(pm)
        step[k] = (cnots(Hjw), cnots(Hjw, True), cnots(Hc), cnots(Hc, True))
        print(f"   {k}     {step[k][0]:5d} / {step[k][1]:5d}              {24 * k * (k - 1) + 2 * k:5d}                "
              f"{step[k][2]:4d} / {step[k][3]:4d}     strings {Hjw.n_strings() - 1} / {Hc.n_strings() - 1}")
        assert step[k][0] == 24 * k * (k - 1) + 2 * k
    for k in (6, 8):
        print(f"   {k}     {24 * k * (k - 1) + 2 * k:5d} (formula)                                         {2 * k * (k - 1):4d} (formula)")

    # the projector of the refinement path: new levels empty
    Pjw, Pc = projector_empty(8, [4, 5, 6, 7]), projector_empty(4, [2, 3])
    print(f"\nProjector on empty new levels, k = 2 -> 4: {Pjw.n_strings() - 1} Z strings, {cnots(Pjw)} CNOTs (Jordan-Wigner); "
          f"{Pc.n_strings() - 1} strings, {cnots(Pc)} CNOTs (compact)")

    # ---- depth against CNOT count: the pair-hopping block and a full step on the simulator
    pm = PairingModel(4, g=1.0)
    Hjw = JordanWigner(8).map_hamiltonian(pm.h, pm.v)
    circ = Circuit(8)
    for kk, c in Hjw.terms.items():
        pauli_rotation(circ, kk, np.real(c) * 0.1)
    ng, n2, depth = gate_counts(circ)
    print(f"\nOne Jordan-Wigner step as a circuit on the simulator: {ng} gates, {n2} CNOTs, depth {depth} "
          f"(without cancelling adjacent basis changes)")

    # ---- the budget, k = 4 (Table 8.depth)
    J = 6                                                   # pair-hopping terms at k = 4
    rng = np.random.default_rng(11)
    sigma, dt = 4.0, 0.25
    mean_steps = np.mean(np.ceil(np.abs(rng.normal(0, sigma, 400000)) / dt))
    print(f"\nBudget at k = 4, N = 4: n_s = 80 second-order refinement steps, rodeo sigma = {sigma}, dt = {dt} "
          f"(mean steps per cycle {mean_steps:.1f})")
    print("                                   Jordan-Wigner (8+1 qubits)     compact (4+1 qubits)")
    diag_jw, diag_c = step[4][0] - J * 48, step[4][2] - J * 4          # ZZ layer; compact has no two-qubit diagonal
    ref_jw = 80 * (second_order(J, 48, diag_jw) + cnots(Pjw))
    ref_c = 80 * (second_order(J, 4, diag_c) + cnots(Pc))
    cdiag_jw, cdiag_c = step[4][1] - J * 64, step[4][3] - J * 8
    cyc_jw = mean_steps * second_order(J, 64, cdiag_jw)
    cyc_c = mean_steps * second_order(J, 8, cdiag_c)
    print(f"  second-order step, plain          {second_order(J, 48, diag_jw):8d}                       {second_order(J, 4, diag_c):6d}")
    print(f"  second-order step, controlled     {second_order(J, 64, cdiag_jw):8d}                       {second_order(J, 8, cdiag_c):6d}")
    print(f"  refinement, 80 steps              {ref_jw:8d}                       {ref_c:6d}")
    print(f"  one rodeo cycle (mean)            {cyc_jw:8.0f}                       {cyc_c:6.0f}")
    for M in (2, 6):
        print(f"  pipeline, M = {M}                   {ref_jw + M * cyc_jw:8.0f}                       {ref_c + M * cyc_c:6.0f}")
    print(f"  UCCSD preparation                 {uccsd_counts(4)[2]:8d}")

    # ---- the other models: one step, plain and controlled (encodings of Table 6.models)
    print("\nOne first-order Trotter step of the four models of Table 6.models")
    print("   model              qubits  strings  CNOTs  controlled")
    for name, H, E0, psi, sector in book_models():
        hI, Ht = split_identity(H)
        print(f"   {name.replace('$', ''):18s} {H.n:4d}   {Ht.n_strings():5d}   {cnots(Ht):5d}   {cnots(Ht, True):6d}")

    # ---- faults
    print("\nExpected faults N eps and success probability exp(-N eps)")
    circuits = (("UCCSD, k = 3", uccsd_counts(3)[2]), ("UCCSD, k = 4", uccsd_counts(4)[2]),
                ("pipeline, compact, M = 2", ref_c + 2 * cyc_c), ("pipeline, JW, M = 2", ref_jw + 2 * cyc_jw))
    eps_list = (1e-2, 1e-3, 1e-4, 1e-6)
    print("   circuit                     CNOTs   " + "   ".join(f"eps={e:.0e}" for e in eps_list))
    for name, n in circuits:
        print(f"   {name:26s} {n:7.0f}   " + "   ".join(f"{np.exp(-n * e):9.2e}" for e in eps_list))
