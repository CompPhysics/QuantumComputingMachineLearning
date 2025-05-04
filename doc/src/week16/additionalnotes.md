### **1. Classical vs. Quantum Boltzmann Machines**  
#### **1.1 Energy-Based Models and Statistical Mechanics**  
Classical BMs use the Gibbs distribution:  
\[
p(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} e^{-E(\mathbf{v}, \mathbf{h})}, \quad Z = \text{Tr}(e^{-E})
\]  
For QBMs, replace classical spins with qubits and use the **quantum Hamiltonian**:  
\[
\hat{H} = -\sum_i h_i \sigma_i^z - \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z - \Gamma \sum_i \sigma_i^x
\]  
where \(\Gamma\) is a transverse field introducing quantum fluctuations.  

#### **1.2 Quantum Thermal States**  
The thermal state \(\rho = \frac{e^{-\beta \hat{H}}}{Z}\) is prepared via imaginary-time evolution. For graduate students, derive \(\rho\) from the Schrödinger equation:  
\[
\frac{\partial \rho}{\partial \beta} = -\frac{1}{2} (\hat{H}\rho + \rho\hat{H})
\]  
This connects to the **Liouville-von Neumann equation** for open quantum systems.  

---

### **2. Mathematical Framework of QBMs**  
#### **2.1 Variational Free Energy**  
The objective function for QBMs minimizes quantum relative entropy:  
\[
\mathcal{F}(\theta) = \text{Tr}[\rho_{\text{model}} \ln \rho_{\text{model}}] - \text{Tr}[\rho_{\text{model}} \ln \rho_{\text{data}}]
\]  
**Gradient Derivation**: For Hamiltonian parameters \(\theta = \{h_i, J_{ij}\}\):  
\[
\frac{\partial \mathcal{F}}{\partial \theta_k} = \beta \left( \langle \hat{H}_k \rangle_{\rho_{\text{data}}} - \langle \hat{H}_k \rangle_{\rho_{\text{model}}} \right)
\]  
Derived using \(\frac{\partial \ln Z}{\partial \theta_k} = -\beta \langle \hat{H}_k \rangle\) and the cyclic property of trace.  

#### **2.2 Training Algorithm**  
1. Prepare \(\rho_{\text{data}}\) using a quantum circuit.  
2. Sample visible units to compute \(\langle \hat{H}_k \rangle_{\text{data}}\).  
3. Update parameters via gradient descent:  
  \[
  \theta_{k+1} = \theta_k - \eta \frac{\partial \mathcal{F}}{\partial \theta_k}
  \]  

---

### **3. Implementing QBMs with PennyLane**  
#### **3.1 Code Example 1: Sampling from a QBM**  
```python
import pennylane as qml
from pennylane import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qbm_circuit(params, beta=1.0):
   h, J = params
   # Apply transverse field and Ising couplings
   for i in range(n_qubits):
       qml.RX(beta * h[i], wires=i)
   for i in range(n_qubits):
       for j in range(i+1, n_qubits):
           qml.IsingZZ(2 * beta * J[i][j], wires=[i, j])
   return qml.probs(wires=range(n_qubits))

# Initialize random parameters
h = np.random.rand(n_qubits)
J = np.random.rand(n_qubits, n_qubits)
probs = qbm_circuit((h, J), beta=2.0)
print("Thermal probabilities:", probs)
```

#### **3.2 Code Example 2: Training a QBM for Stock Price Prediction**  
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load synthetic financial data (e.g., S&P 500 trends)
data = pd.read_csv("stock_data.csv")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values)

# Define QBM for binary price movements (up/down)
def qbm_cost(params, data_samples):
   model_probs = qbm_circuit(params)
   return -np.mean(np.log(model_probs[data_samples]))

# Train using Adam optimizer
opt = qml.AdamOptimizer(stepsize=0.01)
params = (np.random.rand(n_qubits), np.random.rand(n_qubits, n_qubits))

for epoch in range(100):
   data_batch = sample_data_batch(scaled_data)  # Assume implemented
   params, cost_val = opt.step_and_cost(qbm_cost, params, data_batch)
   print(f"Epoch {epoch}, Cost: {cost_val}")
```

---

### **4. Applications in Finance**  
#### **4.1 Portfolio Optimization**  
Map portfolio weights to qubit states. Use QBMs to minimize risk (variance):  
\[
\hat{H}_{\text{portfolio}} = \sum_{i,j} \sigma_i^z \sigma_j^z \Sigma_{ij} - \mu \sum_i \sigma_i^z r_i
\]  
where \(\Sigma_{ij}\) is the covariance matrix and \(r_i\) are returns.  

**Code Example**:  
```python
def portfolio_hamiltonian(returns, cov_matrix, mu=0.1):
   H = 0
   for i in range(n_assets):
       H -= mu * returns[i] * qml.PauliZ(i)
       for j in range(n_assets):
           H += cov_matrix[i][j] * qml.PauliZ(i) @ qml.PauliZ(j)
   return H
```

#### **4.2 Option Pricing**  
Model risk-neutral probabilities for European options. Train a QBM to generate future price paths:  
\[
C(K, T) = e^{-rT} \mathbb{E}_{\text{QBM}}[\max(S_T - K, 0)]
\]  

**Code Example**:  
```python
def option_pricing_qbm(S0, K, r, T, params):
   # Generate price paths using QBM
   final_prices = []
   for _ in range(1000):
       sample = qbm_circuit(params)
       S_T = S0 * (1 + 0.1 * (2*sample[0] - 1))  # Binary up/down
       final_prices.append(np.maximum(S_T - K, 0))
   return np.exp(-r*T) * np.mean(final_prices)
```

---

### **5. Challenges and Quantum Advantage**  
- **NISQ Limitations**: Noise limits qubit count and circuit depth. Use error mitigation (e.g., zero-noise extrapolation).  
- **Entanglement**: Financial time series exhibit temporal correlations. QBMs with entangled hidden units model these better than classical BMs.  

---

### **6. Conclusion**  
QBMs offer a bridge between quantum computing and finance, but require hybrid quantum-classical algorithms for near-term feasibility.  

---

**References**  
1. Orús, R., Mugel, S., & Lizaso, E. (2019). *Quantum computing for finance: Overview and prospects*. Reviews in Physics.  
2. PennyLane Quantum Finance Tutorials: https://pennylane.ai/qml/demos_quantum-finance.html  

**Problem Sets**  
1. Derive \(\partial \mathcal{F}/\partial J_{ij}\) for a 2-qubit QBM.  
2. Implement a QBM to predict Bitcoin volatility using PennyLane.  