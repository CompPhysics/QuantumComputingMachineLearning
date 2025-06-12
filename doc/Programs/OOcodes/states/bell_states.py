import numpy as np
from core.circuit import Circuit, Gate
from core import gates

def bell_state(label="Phi+"):
    c = Circuit(2)
    c.add_gate(Gate(gates.H(), [0]))
    c.add_gate(Gate(gates.CNOT(), [0,1]))

    if label == "Phi+":
        pass
    elif label == "Phi-":
        c.add_gate(Gate(gates.Z(), [0]))
    elif label == "Psi+":
        c.add_gate(Gate(gates.X(), [1]))
    elif label == "Psi-":
        c.add_gate(Gate(gates.X(), [1]))
        c.add_gate(Gate(gates.Z(), [0]))
    else:
        raise ValueError("Unknown Bell state label")

    c.run()
    return c

