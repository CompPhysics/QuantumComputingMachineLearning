
import matplotlib.pyplot as plt
from states.bell_states import bell_state

labels = ["Phi+", "Phi-", "Psi+", "Psi-"]
shots = 1000

for label in labels:
    print(f"\nBell state {label}")
    c = bell_state(label)
    probs = c.get_probabilities()
    print("Statevector probabilities:", probs)
    counts = c.measure(shots=shots)
    print("Measurement counts:", counts)
    plt.bar(counts.keys(), counts.values())
    plt.title(f"Bell state {label}")
    plt.show()

