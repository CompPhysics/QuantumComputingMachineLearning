"""Gate-based statevector simulator and the Pauli-rotation primitive."""
from ._gates import (apply_1q, apply_cnot, Ry, Rz, Rx, H as Hgate, S as Sgate,
                     basis_change, pauli_exp, pauli_matrix)
__all__ = ["apply_1q", "apply_cnot", "Ry", "Rz", "Rx", "Hgate", "Sgate",
           "basis_change", "pauli_exp", "pauli_matrix"]
