from states.bell_states import bell_state
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

labels = ["Phi+", "Phi-", "Psi+", "Psi-"]
shots = 1000

for label in labels:
    print(f"\nBell state {label}")
    c = bell_state(label)
    
    # Run statevector simulation
    state = c.run_statevector()
    print("Statevector:", state)
    
    # Run measurements
    c.add_measure_all()
    counts = c.run_measurement(shots=shots)
    print("Measurement counts:", counts)
    
    # Plot histogram
    plot_histogram(counts, title=f"Bell state {label}")
    plt.show()
