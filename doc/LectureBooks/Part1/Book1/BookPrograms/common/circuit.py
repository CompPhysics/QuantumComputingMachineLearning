"""Chapter 2, Sec. 2.8: gates, circuits and their execution on a StateVector.

A Circuit is an ordered list of Gate objects, read left to right like the
circuit diagrams of the text (Fig. 2.circuits).  ``Circuit.run`` applies the
gates in order to a fresh |0...0> register (or to a given StateVector),
``Circuit.unitary`` multiplies the gate matrices into the full 2^n x 2^n
operator, and ``Circuit.draw`` prints an ASCII version of the diagram.

Shared by every chapter; ``from circuit import Circuit, Gate``.
"""
import numpy as np

import gates as G
from statevector import StateVector


class Gate:
    """A unitary with a name, its matrix and the qubits it acts on.

    For controlled gates `controls` lists the control qubits; the matrix is
    the *uncontrolled* operation on `targets`, and control is added when the
    gate is applied (the gate then acts as |1..1><1..1| (x) U + rest)."""

    def __init__(self, name, matrix, targets, controls=(), params=()):
        self.name = name
        self.matrix = np.asarray(matrix, dtype=complex)
        self.targets = [targets] if isinstance(targets, int) else list(targets)
        self.controls = [controls] if isinstance(controls, int) else list(controls)
        self.params = tuple(params)

    @property
    def qubits(self):
        return self.controls + self.targets

    def full_matrix(self):
        """The (2^k x 2^k) matrix on controls + targets, with control added."""
        U = self.matrix
        for _ in self.controls:
            U = G.controlled(U)
        return U

    def dagger(self):
        return Gate(self.name + "^", self.matrix.conj().T, self.targets, self.controls, self.params)

    def __repr__(self):
        p = "" if not self.params else "(" + ",".join(f"{x:.3g}" for x in self.params) + ")"
        c = f" c={self.controls}" if self.controls else ""
        return f"{self.name}{p} t={self.targets}{c}"


class Circuit:
    """A list of gates on n qubits, applied left to right."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gates = []

    # ----------------------------------------------------- gate shorthands
    def add(self, gate):
        self.gates.append(gate)
        return self

    def x(self, q): return self.add(Gate("X", G.X, q))
    def y(self, q): return self.add(Gate("Y", G.Y, q))
    def z(self, q): return self.add(Gate("Z", G.Z, q))
    def h(self, q): return self.add(Gate("H", G.H, q))
    def s(self, q): return self.add(Gate("S", G.S, q))
    def t(self, q): return self.add(Gate("T", G.T, q))
    def rx(self, theta, q): return self.add(Gate("Rx", G.Rx(theta), q, params=[theta]))
    def ry(self, theta, q): return self.add(Gate("Ry", G.Ry(theta), q, params=[theta]))
    def rz(self, theta, q): return self.add(Gate("Rz", G.Rz(theta), q, params=[theta]))
    def u3(self, th, ph, la, q): return self.add(Gate("U", G.U3(th, ph, la), q, params=[th, ph, la]))
    def cnot(self, c, t): return self.add(Gate("X", G.X, t, controls=c))
    def cx(self, c, t): return self.cnot(c, t)
    def cz(self, c, t): return self.add(Gate("Z", G.Z, t, controls=c))
    def cu(self, U, c, t, name="U"): return self.add(Gate(name, U, t, controls=c))
    def swap(self, a, b): return self.add(Gate("SWAP", G.SWAP, [a, b]))
    def toffoli(self, c1, c2, t): return self.add(Gate("X", G.X, t, controls=[c1, c2]))

    def append(self, other):
        for g in other.gates:
            self.add(g)
        return self

    def inverse(self):
        inv = Circuit(self.n)
        for g in reversed(self.gates):
            inv.add(g.dagger())
        return inv

    @property
    def depth(self):
        """Number of layers when gates on disjoint qubits are run in parallel."""
        busy = [0] * self.n
        for g in self.gates:
            layer = 1 + max(busy[q] for q in g.qubits)
            for q in g.qubits:
                busy[q] = layer
        return max(busy) if busy else 0

    # ------------------------------------------------------------ execution
    def run(self, state=None):
        """Apply the gates to `state` (default |0...0>) and return the state."""
        sv = StateVector(self.n) if state is None else state
        for g in self.gates:
            sv.apply(g.full_matrix(), g.qubits)
        return sv

    def unitary(self):
        """The full 2^n x 2^n matrix of the circuit."""
        U = np.eye(2 ** self.n, dtype=complex)
        for col in range(2 ** self.n):
            sv = StateVector(self.n, np.eye(2 ** self.n)[col])
            U[:, col] = self.run(sv).psi
        return U

    # -------------------------------------------------------------- drawing
    def draw(self):
        """ASCII circuit diagram: one wire per qubit, time running to the right."""
        cols = []
        for g in self.gates:
            col = ["---"] * self.n
            lo, hi = min(g.qubits), max(g.qubits)
            for q in range(lo + 1, hi):
                if q not in g.qubits:
                    col[q] = "-|-"
            for c in g.controls:
                col[c] = "-*-"
            if g.name == "SWAP":
                for q in g.targets:
                    col[q] = "-x-"
            elif g.name == "X" and g.controls:
                col[g.targets[0]] = "(+)"
            else:
                for q in g.targets:
                    col[q] = f"[{g.name}]"
            width = max(len(c) for c in col)
            cols.append([c.center(width, "-") for c in col])
        lines = []
        for q in range(self.n):
            lines.append(f"q{q}: |0>--" + "-".join(c[q] for c in cols) + "--")
        return "\n".join(lines)

    def __repr__(self):
        return f"Circuit({self.n} qubits, {len(self.gates)} gates, depth {self.depth})"


# ------------------------------------------------------------ text circuits
def bell_circuit(kind="Phi+"):
    """The four Bell states from |00>: H on qubit 0, CNOT, then X and/or Z."""
    c = Circuit(2).h(0).cnot(0, 1)
    if kind in ("Psi+", "Psi-"):
        c.x(1)
    if kind in ("Phi-", "Psi-"):
        c.z(0)
    return c


if __name__ == "__main__":
    c = bell_circuit("Phi+")
    print(c)
    print(c.draw())
    print("state:", c.run())
    assert np.allclose(c.unitary()[:, 0], [1, 0, 0, 1] / np.sqrt(2))
    bell = {"Phi+": [1, 0, 0, 1], "Phi-": [1, 0, 0, -1], "Psi+": [0, 1, 1, 0], "Psi-": [0, 1, -1, 0]}
    for kind, v in bell.items():
        psi = bell_circuit(kind).run().psi
        v = np.array(v) / np.sqrt(2)
        assert abs(abs(np.vdot(v, psi)) - 1) < 1e-12, kind
    # circuit identity: SWAP = three CNOTs (Fig. 2.circuits)
    three = Circuit(2).cnot(0, 1).cnot(1, 0).cnot(0, 1)
    assert np.allclose(three.unitary(), G.SWAP)
    print(three.draw())
    # the inverse circuit undoes the circuit
    rnd = Circuit(3).h(0).ry(0.7, 1).cnot(0, 2).rz(1.3, 2).swap(1, 2).t(0)
    assert np.allclose(rnd.append(rnd.inverse()).unitary(), np.eye(8))
    print("circuit checks passed")
