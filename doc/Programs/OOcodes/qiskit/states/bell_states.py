from core.circuit import Circuit
from core.gates import Gates

def bell_state(label="Phi+"):
    c = Circuit(2)
    
    # Create standard Bell basis preparation
    Gates.H(c.qc, 0)
    Gates.CNOT(c.qc, 0, 1)
    
    # Apply additional gates depending on label
    if label == "Phi+":
        pass
    elif label == "Phi-":
        Gates.Z(c.qc, 0)
    elif label == "Psi+":
        Gates.X(c.qc, 1)
    elif label == "Psi-":
        Gates.X(c.qc, 1)
        Gates.Z(c.qc, 0)
    else:
        raise ValueError("Unknown Bell state")

    return c
