"""Chapter 2, Sec. 2.10: superdense coding and quantum teleportation.

Both protocols are run on the state-vector simulator with the Circuit class
of Sec. 2.9.  Superdense coding (Fig. 2.superdense) sends two classical bits
with one qubit; the program checks all four messages.  Teleportation
(Fig. 2.teleport) transfers an unknown qubit state with a Bell pair and two
classical bits; the program checks that Bob's qubit equals Alice's original
state for random inputs and all four measurement outcomes (Eq. 2.telecorr).
Runs in a second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from statevector import StateVector
from circuit import Circuit
from density_matrix import DensityMatrix


# ------------------------------------------------------- superdense coding
def superdense_encode(bits):
    """Alice's local operation on her half of |Phi+> for the message b1 b2."""
    ops = {"00": G.I, "01": G.X, "10": G.Z, "11": G.Z @ G.X}
    return ops[bits]


def superdense_coding(bits):
    """Returns the bit string Bob reads out; should equal `bits`."""
    c = Circuit(2).h(0).cnot(0, 1)             # shared Bell pair, qubit 0 with Alice
    sv = c.run()
    sv.apply(superdense_encode(bits), 0)       # Alice encodes and sends qubit 0
    sv.cnot(0, 1).h(0)                         # Bob decodes with the inverse Bell circuit
    return sv.measure_all(np.random.default_rng(0))


# ------------------------------------------------------------ teleportation
def teleport(psi, rng=None):
    """Teleport the single-qubit state psi from qubit 0 (Alice) to qubit 2 (Bob).

    Qubits 1 and 2 hold the Bell pair.  Returns Bob's density matrix and the
    two classical bits Alice sent."""
    sv = StateVector.product(psi, [1, 0], [1, 0])
    sv.h(1).cnot(1, 2)                         # Bell pair between Alice (1) and Bob (2)
    sv.cnot(0, 1).h(0)                         # Alice's Bell measurement, first the basis change
    m0, _ = sv.measure(0, rng=rng)
    m1, _ = sv.measure(1, rng=rng)
    if m1:                                     # classical corrections, Eq. (2.telecorr)
        sv.x(2)
    if m0:
        sv.z(2)
    rho_bob = DensityMatrix.from_state(sv).partial_trace(2)
    return rho_bob, (m0, m1)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("superdense coding:")
    for bits in ("00", "01", "10", "11"):
        got = superdense_coding(bits)
        print(f"  Alice sends {bits} -> Bob reads {got}")
        assert got == bits
    print("\nteleportation:")
    rng = np.random.default_rng(5)
    counts = {}
    for trial in range(200):
        v = rng.normal(size=2) + 1j * rng.normal(size=2); v /= np.linalg.norm(v)
        rho_bob, bits = teleport(v, rng)
        counts[bits] = counts.get(bits, 0) + 1
        assert np.allclose(rho_bob.rho, np.outer(v, v.conj())), "teleportation failed"
    print("  200 random states teleported exactly; outcomes of Alice's measurement:", dict(sorted(counts.items())))
    # a specific example for the text
    v = np.array([np.cos(0.6), np.exp(0.9j) * np.sin(0.6)])
    rho_bob, bits = teleport(v, np.random.default_rng(1))
    print("  input state", v, "-> Alice's bits", bits, "-> Bob's Bloch vector", rho_bob.bloch_vector())
    circ = Circuit(3).h(1).cnot(1, 2).cnot(0, 1).h(0)
    print("\nteleportation circuit before the measurements:\n" + circ.draw())
