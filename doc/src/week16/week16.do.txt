TITLE: Quantum Computing, Quantum Machine Learning and Quantum Information Theories
AUTHOR: Morten Hjorth-Jensen {copyright, 1999-present|CC BY-NC} at Department of Physics, University of Oslo, Norway
DATE: May 14, 2025

!split
===== Plan for the week of May 12-16 =====
!bblock 
o Quantum Boltzmann Machines: Theory and Implementation
  * Quantum neural networks, wrapping up discussions from last week (see notes from last week at URL:"https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/pub/week15/ipynb/week15.ipynb")
  * Classical Boltzmann  Machines (BMs)
  * Restricted Quantum Boltzmann Machines (RQBM)
  * Training Quantum Boltzmann Machines
  * Practical Implementation with PennyLane
o Summary of course and work on  project 2
!eblock

!split
===== Introduction =====

!bblock
Quantum Boltzmann Machines (QBM extend the
classical Boltzmann machine (a probabilistic neural network) into the
quantum domain. QBMs promise richer representations by leveraging
quantum superposition and entanglement, potentially capturing
correlations that classical models cannot .
!eblock

!bblock
In these notes, we review
first classical Boltzmann machines and restricted Boltzmann machines (RBMs).
Thereafter we
introduce QBMs and their restricted variant (RQBM), discuss training
methods, and illustrate practical implementation using
PennyLane. 
!eblock

!split
===== Classical Boltzmann machines =====


!split
===== Quantum Boltzmann Machines (QBMs) =====


A Quantum Boltzmann Machine (QBM) extends a classical BM by replacing
each binary unit with a qubit and generalizing the energy to a quantum
Hamiltonian.


Quantum Boltzmann machines (QBMs) generalize this framework by
encoding the model distribution in a quantum Gibbs state.  Instead of
a classical energy, one defines a Hamiltonian $H(\boldsymbol{\Theta})$
whose parameters $\boldsymbol{\Theta}$ (biases and couplings) play the
role of the RBM weights.  The model’s density operator is the state
!bt
\[
\rho(\boldsymbol{\theta}) = \frac{\exp{-(\beta H(\boldsymbol{\Theta}))}}{Z(\boldsymbol{\Theta})},
\]
!et
with inverse temperature $\beta$ (set to 1 as we did for standard Boltzmann machines ) and partition function
$Z = \Tr(\exp{-\beta H})$.  The probability of observing a visible
configuration v is obtained by measuring $\rho$ in the computational
basis (and tracing out hidden qubits if any).  In effect, the quantum
model can capture richer correlations via superposition and
entanglement.



!split
===== Quantum Boltzmann Machines =====





In a Quantum Boltzmann Machine (QBM), the classical energy is replaced
by a Hamiltonian H acting on qubits.  The model distribution over
classical bitstrings v is given by the diagonal of the quantum Gibbs
state $\rho = e^{-H}/Z$.  A straightforward choice is a stoquastic
Hamiltonian that is diagonal in the computational basis
(e.g. involving only Pauli-$Z$ operators), which yields a probability
distribution very similar to a classical BM.  More generally one can
allow non-commuting terms (e.g. Pauli-$X$ fields) to introduce quantum
correlations .  In fact, Amin et al. (2018) introduced a QBM where the
training is done by bounding the quantum probabilities and sampling
from the transverse-field Ising Hamiltonian .  However,
non-commutativity makes exact training harder, so many proposals use
either special Hamiltonians or variational approximations.

!split

!split
===== Restricted QBM (RQBM) =====

A Restricted Quantum Boltzmann Machine (RQBM) (also called Quantum RBM
or QRBM) enforces a bipartite structure analogous to the classical
RBM: no hidden-hidden interactions, and possibly limited
hidden-visible connectivity.  The simplest RQBM Hamiltonian can be
written (up to local Pauli bases) as
!bt
\[
H(\mathbf{a},\mathbf{b},W,V) \;=\; \sum_{i=1}^{n_v} a_i Z_i \;+\; \sum_{j=1}^{n_h} b_j Z_j \;+\; \sum_{i,j} W_{ij}\, Z_i Z_j \;+\; \sum_{i<i{\prime}} V_{ii{\prime}}\, Z_i Z_{i{\prime}} \,.
\]
!et
Here $Z_i$ and $Z_j$ are Pauli-$Z$ operators on the visible and hidden
qubits respectively, $a_i,b_j$ are biases, $W_{ij}$ are visible-hidden
couplings, and V_{ii{\prime}} are possible visible-visible couplings.
(Classically, V=0 in an RBM; allowing V\neq0 gives a “2-local QRBM” as
in Wu et al. .)  Importantly, there are no hidden-hidden $ZZ$ terms in
this restricted model.  Equation (\ref{eq:rqbm_hamiltonian}) is a
direct quantum analogue of the RBM energy function, promoting it to an
operator acting on qubits.  Wu et al. (2020) used such a Hamiltonian
and proved that this 2-local QRBM is universal for quantum computation
.




!split
===== Quantum Statistical Mechanics Background =====





In quantum statistical mechanics, a system at inverse temperature
\beta is described by the density operator \rho = e^{-\beta H}/Z,
where the partition function \(Z = \Tr(e^{-\beta H})\) normalizes the
state.  For a qubit model we typically take \beta=1.  Observables are
expectation values \(\langle O\rangle = \Tr(\rho O)\). In the QBM
context, one is interested in the probability p(v) of measuring the
visible qubits in computational basis state v.  If the full thermal
state lives on both visible and hidden qubits, this probability is

!bt
\[
p_\theta(v) \;=\; \Tr\bigl[\Pi_v^{(\text{vis})}\,\rho(\theta)\bigr],
\]
!et
where \(\Pi_v^{(\text{vis})}=|v\>\<v|\) acts on the visible subspace.
Equivalently, one may “trace out” the hidden qubits and work with the
reduced density matrix on the visible subsystem.  Computing these
probabilities requires preparing or approximating the Gibbs state of
H.  In practice this is done either by quantum simulators, quantum
annealers, or variational algorithms.




!split
===== Energy-Based Training Objective and Gradients =====


RQBM training is analogous to the classical case: we have a dataset of
bitstrings \{v^{(k)}\} from an unknown distribution p_{\rm data}(v).
The goal is to adjust the Hamiltonian parameters \theta so that the
model distribution \(p_\theta(v)=\<v|\rho(\theta)|v\>\) approximates
p_{\rm data}(v).  Equivalently, one can view the data distribution as
a target density matrix \eta (diagonal in the computational basis) and
minimize the quantum relative entropy (quantum KL divergence)

!bt
\[
S(\eta\Vert \rho(\theta)) = \Tr\!\bigl[\eta\ln\eta\bigr] - \Tr\!\bigl[\eta\ln\rho(\theta)\bigr] \;.
\]
!et

This loss is non-negative and equals zero only when \eta=\rho(\theta).
Writing \rho=e^{-H}/Z, one finds the gradient of the relative entropy
(for parameter \theta in H) as
!bt
\[
\frac{\partial}{\partial\theta} S(\eta\Vert\rho)
= \Tr\!\Bigl[\eta\,\partial_\theta(\beta H + \ln Z)\Bigr]
= \beta\Bigl(\Tr[\eta\,\partial_\theta H] - \Tr[\rho\,\partial_\theta H]\Bigr).
\]
!et
In other words,
!bt
\[
\nabla_\theta S \;=\; \beta\Bigl(\langle \partial_\theta H\rangle_{\rm data} \;-\; \langle \partial_\theta H\rangle_{\rm model}\Bigr).
\]
!et

This is directly analogous to the classical RBM gradient: the update
for each parameter is proportional to the difference between its
expectation under the data distribution and under the model’s Gibbs
distribution.  (Equation: $S(\eta\Vert\rho)={\rm Tr}[\eta\ln\eta]-{\rm
Tr}[\eta\ln\rho]$ .)  In practice, one computes $\langle
\partial_\theta H\rangle_{\rm data}$ by averaging over the training
set, and estimates $\langle \partial_\theta H\rangle_{\rm model}$ by
sampling from the quantum model.



Note that preparing exact Gibbs samples of a non-commuting Hamiltonian
is hard.  Many methods have been proposed to approximate the model
expectation.  For example, one may use a bound on the quantum free
energy (as in Amin et al. ), or perform contrastive divergence with a
quantum device.  Recent theoretical work shows that minimizing the
relative entropy in QBM training can be done with stochastic gradient
descent in polynomial sample complexity under reasonable assumptions .



Figure: Quantum vs. classical training loop for RBMs.  In the classical loop (red arrows), the RBM weights \Theta=(W,a,b) are updated via Gibbs sampling: visible data is clamped and Gibbs steps estimate the “positive phase,” while separate Gibbs chains estimate the “negative phase.”  In the quantum loop (blue arrows, e.g. using a D-Wave annealer), the model samples for the negative phase are drawn by encoding \Theta in the quantum device and measuring its thermal state .  The positive phase is computed classically from the data.  This figure (from Moro et al., 2023) highlights that the only difference is how the negative-phase samples are obtained .




!split
===== Parameter Optimization and Variational Techniques =====

Given the gradient above, one can optimize \theta by standard
gradient-based methods (SGD, Adam, etc.).  In a gate-based setting, we
implement the RQBM Hamiltonian via a parameterized quantum circuit
(ansatz) and use variational quantum algorithms (VQAs).  Each
parameter in H is encoded as a gate angle or circuit parameter.  The
gradient of a circuit expectation can be obtained by the
parameter-shift rule or automatic differentiation.



One approach is the $\beta$-Variational Quantum Eigensolver (β-VQE)
technique.  Liu et al. (2021) proposed a variational ansatz to
represent a thermal (mixed) state using a combination of a classical
neural network and a quantum circuit.  Huijgen et al. (2024) applied
this to QBM training: an inner loop runs β-VQE to approximate the
Gibbs state of H(\theta), while an outer loop updates \theta to
minimize the relative entropy to the data .  This “nested loop”
algorithm effectively sidesteps direct sampling of the true quantum
Boltzmann state by variational approximation.  It has been shown to
work on both classical and quantum target data, achieving
high-fidelity learning for up to 10 qubits .



Other sophisticated ansätze exist.  For example, Evolved Quantum
Boltzmann Machines (Minervini et al., 2025) prepare a thermal state
under one Hamiltonian and then evolve it under another, combining
imaginary- and real-time evolution.  They derive analytical gradient
formulas and propose natural-gradient variants .  There are also
“semi-quantum” RBMs (sqRBMs) which commute in the visible subspace and
treat the hidden units quantum-mechanically.  Intriguingly, sqRBMs
were found to be expressively equivalent to classical RBMs, requiring
fewer hidden units for the same number of parameters . In practice,
however, variational optimization in high dimensions can suffer from
barren plateaus. Recent analysis shows that training QBMs using the
relative entropy objective avoids many of these concentration issues,
with provably polynomial complexity under realistic conditions .




!split
===== Implementation with PennyLane =====



As a concrete example, we outline how to implement an RQBM in
PennyLane. We consider n_v visible and n_h hidden qubits. The ansatz
can be, for instance, layers of parameterized single-qubit rotations
and entangling gates that respect the bipartite structure.  Below is
illustrative code (in Python) using PennyLane’s default.qubit
simulator.

!bc pycod
import pennylane as qml
import numpy as np
# Number of visible and hidden qubits
n_v, n_h = 2, 1
dev = qml.device(“default.qubit”, wires=n_v+n_h)

# Define a variational circuit (QNode)

@qml.qnode(dev, interface=‘autograd’)
    def circuit(params):

# params is a vector of rotation angles

# Prepare all qubits in |0>

# Example ansatz: one layer of rotations + entangling gates
    for i in range(n_v + n_h):
        qml.RY(params[i], wires=i)
# entangle visible to hidden
        for i in range(n_v):
            qml.CNOT(wires=[i, n_v])  # connect each visible i to hidden n_v

# Optionally more layers…

# Return probability distribution on visible wires

return qml.probs(wires=list(range(n_v)))
!ec



This circuit takes a parameter vector params of length n_v+n_h and returns the probabilities q_\theta(v) of measuring each visible bitstring v.  Notice we measure only the visible wires (the wires=list(range(n_v)) in qml.probs marginalizes out the hidden qubit).



Next, we train this model to match a target dataset distribution.
Suppose our data has distribution target = [p(00), p(01), p(10),
p(11)].  We can define the (classical) loss as the Kullback-Leibler
divergence D_{\rm KL}(p_{\rm data}\Vert q_\theta) or simply the
negative log-likelihood.  Then we update params by gradient descent.
PennyLane’s automatic differentiation can compute gradients via the
parameter-shift rule, but we show an explicit parameter-shift
computation for demonstration:

!bc pycod
# Example target distribution over 2 visible bits
target = np.array([0.3, 0.2, 0.1, 0.4])  # must sum to 1
def loss(params):
    probs = circuit(params)  # model probabilities for visible states
# Add small epsilon to avoid log(0)
    return np.sum(target * np.log((target + 1e-9) / probs))





Compute gradient via parameter-shift rule





def parameter_shift_grad(params):

grads = np.zeros_like(params)

shift = np.pi/2

for idx in range(len(params)):

shift_vector = np.zeros_like(params)

shift_vector[idx] = shift

probs_plus = circuit(params + shift_vector)

probs_minus = circuit(params - shift_vector)

loss_plus  = np.sum(target * np.log((target + 1e-9) / probs_plus))

loss_minus = np.sum(target * np.log((target + 1e-9) / probs_minus))

grads[idx] = 0.5 * (loss_plus - loss_minus)

return grads





Initialize parameters and perform a simple gradient descent





params = np.random.normal(0, 0.1, size=(n_v+n_h,))

learning_rate = 0.1

for epoch in range(100):

grads = parameter_shift_grad(params)

params -= learning_rate * grads

\end{lstlisting}



This code illustrates the training loop.  At each epoch we evaluate the loss, compute gradients (via two forward passes per parameter), and update the parameters.  In practice one can use qml.grad for automatic gradients, and more sophisticated optimizers (Adam, natural gradient, etc.). The above shows that PennyLane can seamlessly integrate quantum circuit definitions with classical training logic.





Applications and Examples





RQBMs and related quantum generative models have begun to find applications in unsupervised learning and physics.  For example, anomaly detection in cybersecurity can be cast as a generative modeling task: anomalies are rare samples of a complex distribution.  Stein et al. (2023) built fully unsupervised anomaly detectors using QBMs trained on synthetic intrusion data.  Their results indicate that, for certain tasks, the quantum model can achieve better anomaly-detection performance than the classical RBM, often requiring fewer training steps .  Likewise, Moro & Prati (2023) demonstrated a quantum speed-up in training RBMs on a D-Wave annealer: the negative-phase sampling was up to 64× faster in hardware than in a CPU implementation for real-world datasets, although overheads remain .



In another large-scale example, Sinno et al. (2025) implemented a QRBM with 120 visible and 120 hidden units on D-Wave’s Pegasus chip to generate synthetic network traffic for intrusion detection.  They successfully generated over 1.6 million attack samples, achieving a balanced dataset of more than 4.2 million records.  Compared to classical oversampling methods (SMOTE, etc.), the QRBM-generated data led to higher detection rates and F1 scores across multiple classifiers . These results highlight the potential of RQBMs as quantum generative models in practical machine-learning workflows.



RQBMs are also studied in physics.  For instance, Wu et al. (2020) used a QRBM ansatz on a superconducting quantum chip to approximate quantum wavefunctions. They trained the QRBM so that its output state approximated the ground state and Gibbs (thermal) state of small molecules, achieving reasonable accuracy .  This connects to the broader idea of using neural-network quantum states (like RBMs) to represent many-body wavefunctions. Indeed, Carleo & Troyer (2017) introduced classical RBMs as variational ansätze for quantum ground states, and unitary/complex extensions have been explored in many works . A Restricted Quantum Boltzmann Machine provides an alternative ansatz where part of the network is truly quantum; recent theory shows that allowing non-commuting terms (e.g. in the hidden layer) does not expand representational power beyond classical RBMs, though it does change resource requirements .



Overall, RQBMs sit at the intersection of quantum statistical physics and machine learning. They generalize RBMs to the quantum domain, require understanding of density matrices and partition functions, and employ energy-based learning objectives analogous to classical models. Variational training techniques and software frameworks like PennyLane make it possible to experiment with RQBMs on near-term devices. As research progresses, we expect RQBMs and related models to be applied to more sophisticated generative tasks and potentially offer advantages in sampling, expressivity, or training speed .





References





Amin et al., 2018. Quantum Boltzmann Machine, Phys. Rev. X 8, 021050. (Introduced the QBM and training bounds .)
Wu et al., 2020. Quantum restricted Boltzmann machine is universal for quantum computation, arXiv:2005.11970. (Defined the 2-local QRBM Hamiltonian and demonstrated its universality .)
Huijgen et al., 2024. Training Quantum Boltzmann Machines with the β-VQE, arXiv:2304.08631. (Presented the nested variational training algorithm .)
Coopmans & Benedetti, 2024. On the sample complexity of quantum Boltzmann machine learning, Commun. Phys. 7, 274. (Theoretical analysis of relative-entropy training and sample complexity .)
Minervini et al., 2025. Evolved Quantum Boltzmann Machines, arXiv:2501.03367. (Proposed the eQBM ansatz mixing imaginary and real time evolution .)
Nicosia et al., 2025. Expressive equivalence of classical and quantum RBMs, arXiv:2502.17562. (Introduced semi-quantum RBMs (sqRBMs) with commuting visible terms and non-commuting hidden terms; showed structural relationships with classical RBMs .)
Stein et al., 2023. Unsupervised anomaly detection with Quantum Boltzmann Machines, IEEE QWeek (preprint arXiv:2306.04998). (Applied QBMs to fraud/anomaly detection; found QBMs could outperform classical RBMs on synthetic cybersecurity data .)
Moro & Prati, 2023. Anomaly detection speed-up by quantum restricted Boltzmann machines, Commun. Phys. 6, 269. (Demonstrated classical vs. quantum training loops on real datasets and observed large sampling speed-ups on a quantum annealer .)
Sinno et al., 2025. Implementing Large Quantum Boltzmann Machines for Dataset Balancing, arXiv:2502.03086. (Embedded a 120×120 QRBM on D-Wave Pegasus to generate millions of intrusion-detection samples, improving downstream classifier performance .)




!bt
\[
H = -\sum_{a=1}^N \Gamma_a,\sigma^x_a ;-;\sum_{a=1}^N b_a,\sigma^z_a ;-;\sum_{a<b} u_{ab},\sigma^z_a \sigma^z_b,
\]
!et

where $\sigma_a^x,\sigma_a^z$ are the Pauli matrices on qubit $a$,
$\Gamma_a$ is a “transverse” field, $b_a$ are local fields (biases),
and $u_{ab}$ are interaction strengths .  Here each qubit can be
interpreted analogously to a classical unit, but the $\sigma^x$ term
induces quantum superposition.  One can also include $\sigma^y$ or
more complex terms, but we focus on this common ansatz.







The quantum model assigns a density matrix (mixed state) to the system via the Gibbs (thermal) state at some inverse temperature $\beta=1/T$:

$$

\rho ;=; \frac{e^{-\beta H}}{\Tr(e^{-\beta H})},.

$$

Measuring $\rho$ in the computational ($\sigma^z$) basis yields a probability distribution over bit-strings $z\in{0,1}^N$:

$$

p(z) = \bra{z} \rho \ket{z} ;=; \frac{\bra{z} e^{-\beta H} \ket{z}}{\Tr(e^{-\beta H})}.

$$

This model is a quantum generalization of the Boltzmann distribution: if $H$ were diagonal in the $z$-basis (i.e.\ $\Gamma_a=0$), then $\bra{z} e^{-\beta H} \ket{z} = e^{-\beta E(z)}$ and one recovers a classical BM. But with nonzero $\Gamma$, $H$ has off-diagonal terms, and $\rho$ generally has entanglement and non-commuting contributions.



One advantage of QBMs is expressivity: quantum Hamiltonians can capture correlations that classical ones cannot. For instance, the non-commuting $\sigma^x$ terms allow exploring multiple states in superposition. Studies suggest that even when modeling classical data, QBMs can outperform classical BMs in capturing complex patterns .  Askarzadeh et al. note that a QBM can include “more general non-commuting terms,” making it strictly more expressive .  In practice, one often treats the visible data as classical: one may fix (clamp) the “visible” qubits to data values and only allow quantum sampling over hidden qubits, similar to RBMs. But fundamentally, a QBM defines a quantum Boltzmann distribution as above .



In summary, a QBM is a physics-inspired generative model: its parameters $(\Gamma, b, u)$ define a Hamiltonian, and sampling is done by preparing the Gibbs state of that Hamiltonian. By training these parameters, the QBM is intended to learn to generate data similar to the training set.  As with classical BMs, one can consider visible and hidden qubits: visible qubits may be initialized to classical data (e.g.\ via computational-basis initialization), while hidden qubits are free to evolve quantumly.  The model is trained so that measurements on the visible qubits (after tracing out or measuring hidden qubits) reproduce the data distribution .



Why quantum?  Intuitively, using a quantum system can allow more compact representations.  For example, Gurvits and others have shown that quantum statistics can represent certain probability distributions more efficiently than classical spins.  Moreover, upcoming quantum hardware (e.g.\ quantum annealers) may naturally implement sampling from such quantum Boltzmann distributions.  While the ultimate advantage is still under research, the idea is that QBMs could be components of future quantum machine learning systems, possibly providing speedups or capturing quantum correlations in quantum data .



Representative fact:  Tan et al. (2023) remark that a QBM is “a physically motivated quantum neural network” for generative/discriminative tasks . Similarly, Amin et al. (2018) introduce a QBM learning algorithm and demonstrate via simulations that “the machine exploits its quantum nature to mimic data sets” .



Exercises (Chapter 3): (1) QBM Hamiltonian: Write the Hamiltonian $H$ for a QBM with two visible qubits and one hidden qubit, using bias parameters $(b_1,b_2)$ for visibles, $(c_1)$ for hidden, weights $W_{1},W_{2}$ between visible 1 and hidden, etc., plus transverse fields $\Gamma_i$. (2) Quantum vs Classical: In the limit $\Gamma_a\to 0$, show that the QBM reduces to a classical BM. What role does $\Gamma$ play? (3) Density matrix: Explain why the thermal state $\rho=e^{-\beta H}/Z$ is a density matrix and how measuring in the computational basis yields a probability distribution over bit-strings.





Restricted Quantum Boltzmann Machines (RQBM)





As in the classical case, one can restrict a QBM’s connectivity to simplify training. A Restricted QBM (RQBM) has a bipartite graph structure: visible qubits connect only to hidden qubits, with no interactions among visibles or among hiddens.  The Hamiltonian then has couplings $u_{v,h}$ only between a visible qubit $v$ and a hidden qubit $h$.  Formally, for $n$ visible and $m$ hidden qubits, one can write:

$$

H = -\sum_{i=1}^n b_i,\sigma^z_{v_i} - \sum_{j=1}^m c_j,\sigma^z_{h_j} - \sum_{i,j} W_{ij},\sigma^z_{v_i}\sigma^z_{h_j} ;-; \sum_{j=1}^m \Gamma_j,\sigma^x_{h_j},

$$

where we may allow transverse fields only on the hidden qubits (visible qubits are clamped to data and treated classically). This is a quantum analog of the classical RBM: visible units (qubits) have no $\sigma^z$–$\sigma^z$ interactions among themselves, only with hidden qubits.



Such an architecture can make training easier.  Wiebe and Wossnig (2019) specifically consider a class of RQBMs where the hidden-qubit Hamiltonians commute, allowing certain variational training techniques . They point out that classical pre-training or other bounds can be used in the restricted case.  In reinforcement learning applications, Amin et al. have applied an RQBM to learn policies by clamping visible inputs (state-action pairs) and letting hidden qubits represent the “value” function .



In general, the RQBM still defines a quantum Boltzmann distribution: the Gibbs state of the above Hamiltonian yields

$$

\rho = \frac{e^{-\beta H}}{\Tr e^{-\beta H}}, \qquad

P(v) = \Tr_{h}\bigl[\bra{v} \rho \ket{v}\bigr],

$$

where $\Tr_h$ denotes tracing over hidden qubits. Training then adjusts ${W,b,c,\Gamma}$ so that $P(v)$ matches the data (where $v$ runs over visible bit-strings). By enforcing the bipartite connectivity, one hopes that sampling and gradient estimates can be done more efficiently, akin to classical RBM training.



Analogy: The RQBM retains the spirit of a classical RBM but with quantum dynamics on the hidden layer. One can view it as a hybrid: the visible qubits are often “classical” (fixed in the computational basis for each data point), while hidden qubits evolve under a transverse-field Ising Hamiltonian. This structure has been used both theoretically and experimentally to realize QBMs on hardware like quantum annealers .



Exercises (Chapter 4): (1) Graph structure: Draw an RQBM with 2 visible and 2 hidden qubits, labeling the Hamiltonian terms (biases on all qubits and weights only between visible–hidden pairs). (2) Commuting hidden Hamiltonians: Why might one require hidden-qubit Hamiltonians to commute for certain training algorithms? Discuss the case where $\Gamma$ terms are equal on all hidden qubits. (3) Connection to quantum annealing: Explain how an RQBM Hamiltonian is similar to the Hamiltonians used in quantum annealing devices (e.g.\ D-Wave), and why annealers naturally sample from (or approximate) such quantum Boltzmann distributions.





Training Quantum Boltzmann Machines





Training a QBM means optimizing its parameters so that the model distribution $\rho$ fits the data distribution. A natural objective is the quantum relative entropy (quantum Kullback–Leibler divergence) between the data ensemble and the model’s Gibbs state. Equivalently, one minimizes the negative log-likelihood of the data under the QBM.  The gradient of this loss with respect to a parameter $\theta$ typically involves terms like $\partial_\theta \Tr(\rho \ln \rho_D)$, which translate to differences of expectation values. For a QBM with Hamiltonian $H(\theta)$, it can be shown that the gradient of the log-likelihood leads to updates of the form :

$$

\delta \theta ;\propto; \langle O_\theta \rangle_{\text{data}} ;-; \langle O_\theta \rangle_{\text{model}},

$$

where $O_\theta = \partial_\theta H$ is the operator conjugate to $\theta$, and the expectations are taken over the data-clamped and model Gibbs states respectively. For example, for a weight $W_{ij}$ the gradient involves $\langle \sigma^z_{v_i}\sigma^z_{h_j}\rangle$, and for a bias $b_i$ it involves $\langle\sigma^z_{v_i}\rangle$.



However, unlike the classical case, computing the model expectations $\langle O_\theta\rangle_{\text{model}}=\Tr(\rho,O_\theta)$ requires sampling from a quantum thermal state $\rho=e^{-\beta H}/Z$.  Preparing such a Gibbs state on a quantum computer is generally hard (in fact NP-hard in worst case) .  As Tan et al. note, training requires many such samples since each gradient estimate demands potentially exponential resources .



Thus, training QBM is challenging. Several approaches exist:



Quantum variational training: One can use a variational quantum circuit to approximate the Gibbs state. For example, Quantum Imaginary Time Evolution (QITE) or a layered ansatz can approximate $e^{-\beta H}\ket{+}^{\otimes n}$. The parameters of this circuit are then optimized so that its output matches data statistics.  PennyLane and other frameworks can compute gradients via parameter-shift rules for such circuits. (This approach requires a fault-tolerant device or clever heuristics.)
Bounds and approximations: Amin et al. (2018) circumvented the non-commutativity issue by introducing a bound on quantum probabilities, allowing approximate sampling . In their method, they derive a lower bound to the log-likelihood that can be sampled efficiently. They demonstrate training with and without this bound, using exact diagonalization on small systems.  The essence is to replace $e^{-\beta H}$ with an operator that is easier to sample from, controlling the error.
Restricted/commuting methods: Wiebe & Wossnig (2019) developed methods for Restricted QBMs with commuting hidden Hamiltonians. By ensuring that the hidden-qubit Hamiltonian terms commute, one can derive a variational upper bound on the quantum relative entropy and optimize it classically .  In this restricted case, the problem simplifies, and gradient evaluation can be done using linear combinations of unitaries or by sampling from simpler Hamiltonians.
Hybrid annealing: One proposal is to use quantum annealers themselves as Boltzmann samplers. If one encodes the QBM Hamiltonian (with quantum fluctuations) onto a quantum annealer, then running it at finite temperature might (approximately) sample from the needed quantum Gibbs distribution. Some works explore training annealer qubits to mimic a QBM through reverse annealing or similar techniques .
Coreset and data reduction: Recently, Tan et al. (2023) suggested using coresets (small representative data subsets) to reduce training cost . The idea is to compress the training set into a weighted coreset so that the QBM only needs to match a small number of effective data points, reducing the number of Gibbs-state preparations needed.




Overall, training a QBM often involves computing gradients of the form

$$

\partial_\theta \bigl[-\log \Tr(e^{-\beta H})\bigr] ;=; \langle \partial_\theta H \rangle_{\text{model}}!,

$$

which is then set against empirical averages.  In practice, one may replace exact gradients with approximate or stochastic estimates.  Note that when $\Gamma=0$, the model is classical and one recovers standard BM gradients .



Contrast with classical training:  For classical RBMs, the Contrastive Divergence (CD) algorithm provides an efficient approximate gradient by a short Gibbs chain . A quantum analog of CD is less straightforward, due to the need to sample a quantum state. Some proposals use one round of (partial) quantum annealing as the “negative phase” to get an estimate of model correlations. These methods are still experimental.



Exercise: Show that minimizing the quantum relative entropy $S(\rho_D|\rho_\theta)$ (where $\rho_D$ is the data density matrix and $\rho_\theta=e^{-\beta H(\theta)}/Z$) is equivalent to maximizing the log-likelihood of the data under the model. Derive the gradient $\partial_\theta S$ to confirm it yields terms of the form $\langle \partial_\theta H \rangle_{\text{data}} - \langle \partial_\theta H \rangle_{\text{model}}$.





Practical Implementation with PennyLane





To illustrate QBM concepts, we can use PennyLane – a Python library for quantum machine learning – to implement simple generative models. Below we sketch a toy example: a 2-qubit Quantum Circuit Born Machine (QCBM), which generates a probability distribution via a parameterized quantum circuit. While a QCBM is not exactly a QBM (it has no thermal state), it serves to show how to code a quantum generative model.

import pennylane as qml
from pennylane import numpy as np

# Use a 2-qubit simulator
dev = qml.device('default.qubit', wires=2)

# Define a simple 2-qubit QCBM circuit
@qml.qnode(dev)
def circuit(params):
    # params = 4 angles
        qml.RX(params[0], wires=0)
	    qml.RY(params[1], wires=1)
	        qml.CNOT(wires=[0, 1])
		    qml.RX(params[2], wires=0)
		        qml.RY(params[3], wires=1)
			    return qml.probs(wires=[0,1])  # return probabilities of |00>,|01>,|10>,|11>

# Initialize random parameters
params = np.random.randn(4, requires_grad=True)

# Get the output distribution from the circuit
probs = circuit(params)
print("Probabilities:", probs)
In this code, we create a 2-qubit variational circuit and return the probabilities of each basis state. One could train params to match a target distribution by defining a cost (e.g. KL divergence) between probs and the data distribution, and using PennyLane’s automatic differentiation to update params.  Though not a true QBM (no Hamiltonian or Gibbs state), this demonstrates how one might build and train a small quantum generative model in PennyLane.



To make it more QBM-like, one could encode the classical visible units as fixed inputs. For example, to compute the energy of a QBM Hamiltonian for a given visible bitstring, one might do:

# Example: compute expectation of a simple Hamiltonian for a given state |v>
H = 1.5 * qml.PauliZ(0) + 0.7 * qml.PauliZ(1) + 0.9 * qml.PauliZ(0)@qml.PauliZ(1)

@qml.qnode(dev)
def energy_of_state():
    # Prepare visible qubits in state |v> = |01>, say
        qml.PauliX(wires=1)  # flips qubit 1 to |1>
	    # No hidden qubits in this simple example
	        # Return expectation of H in this state
		    return qml.expval(H)

print("Energy of |01> state:", energy_of_state())
Here, we used a 2-qubit Hamiltonian $H=1.5\sigma_z^0+0.7\sigma_z^1+0.9\sigma_z^0\sigma_z^1$ and prepared the state $|v>=|01>$. The expval(H) call returns $\langle 01|H|01\rangle = -1.5 + 0.7 - 0.9 = -1.7$ (up to sign conventions). This shows how PennyLane can compute energies of basis states, which is a key step in evaluating the QBM energy and probability of data states.



In practice, training a QBM in PennyLane would involve preparing quantum states corresponding to the Gibbs distribution. One could use functions like qml.qaoa.cost.Hamiltonian or custom circuits to approximate $e^{-\beta H}$, and then use PennyLane’s optimizers. Also, PennyLane can interface with PyTorch or TensorFlow, enabling hybrid optimization of quantum circuits with classical parameters (e.g. the Hamiltonian weights).



Summary of steps in PennyLane implementation:



Define a quantum device (qml.device) with enough wires for visibles+hidden.
Construct a parametric quantum circuit (ansatz) that depends on trainable parameters (e.g. angles in rotations and entangling gates).
If modeling an RQBM, one might clamp visible qubits (preparing them according to training data) and apply gates only to hidden qubits.
Measure relevant observables: one can return probabilities (probs) or expectation values (expval) of Pauli operators, depending on the chosen loss function.
Define a cost function, such as the negative log-likelihood or a distance between output and target distribution.
Use an optimizer (e.g. gradient descent, Adam) with PennyLane’s gradient calculations to update parameters.




Below is a skeleton illustrating training a QCBM on a simple 2-point dataset. This is pseudocode for illustration only:

data = np.array([[0,1], [1,0]])  # two data points (|01> and |10>)

def cost(params):
    loss = 0
        for v in data:
	        # clamp visible by preparing state |v>
		        @qml.qnode(dev)
			        def circuit_clamped():
				            if v[0]==1: qml.PauliX(wires=0)
					                if v[1]==1: qml.PauliX(wires=1)
							            # apply layers on all qubits (visible hidden together in this toy example)
								                qml.RY(params[0], wires=0)
										            qml.RY(params[1], wires=1)
											                return qml.probs(wires=[0,1])
													        prob = circuit_clamped()
														        # compute negative log prob of data state (this is a toy loss)
															        idx = v[0]*2 + v[1]  # index of bitstring in probs
																        loss += -np.log(prob[idx] + 1e-6)
																	    return loss / len(data)

opt = qml.GradientDescentOptimizer(stepsize=0.1)
for step in range(50):
    params, curr_cost = opt.step_and_cost(cost, params)
        if step%10==0:
	        print(f"Step {step}: cost = {curr_cost:.3f}")
		In this toy loop, we optimize the circuit so that the probability of the given data points is high. In a true QBM, the cost would be derived from the log-likelihood of all data under the quantum thermal model.



Exercises (Chapter 5): (1) PennyLane circuit: Modify the above QCBM code to use 3 qubits and train to model a three-bit data distribution. (2) Hamiltonian expectation: Using PennyLane, compute the expectation value of $H = \sigma^x\otimes\sigma^x + \sigma^z\otimes\sigma^z$ on the Bell state $(|00>+|11>)/\sqrt{2}$. (3) Parameter shift rule: Explain how PennyLane can compute gradients of quantum circuit parameters via the parameter-shift rule, and why this is useful for QBM training.





Applications and Examples





Although QBMs are still mainly a research topic, we briefly mention some toy applications and encourage experimentation with small datasets.  Classical RBMs have been used for simple generative tasks like the Bars-and-Stripes dataset (binary images) or the MNIST digit dataset.  QBMs could similarly be tested on these tasks to see if they offer any advantage. For example, one can train a small QBM to generate $4\times4$ binary patterns.  Tan et al. (2023) used QBMs on the Bars-and-Stripes task (4×4 images) as a benchmark .



In reinforcement learning, QBM-like models have been used to learn policies: one treats (state,action) pairs as inputs and uses the QBM to approximate a Q-function .  An RQBM with input neurons (for state), output neurons (for action), and hidden qubits can learn the value of each action by minimizing a “free energy” (analogous to Q-value).  This demonstrates how QBMs could serve as function approximators.



Small dataset example: Consider a 4-bit dataset consisting of the two strings 0000 and 1111. A classical RBM with sufficient hidden units can exactly model this. A QBM with 4 visible qubits and some hidden qubits could also learn it. One could set up the QBM Hamiltonian and run a variational algorithm so that $|0000>$ and $|1111>$ have highest probability.  Training could be done by maximizing $p(0000)+p(1111)$. This is an example of a bimodal distribution that a BM learns.



Another toy example is the XOR problem: classify inputs $[0,0],[0,1],[1,0],[1,1]$ with target labels given by XOR. A small QBM or RQBM could be used as a generative model over the four inputs and supervised labels, similar to a discriminative model. Although this is artificial, it illustrates how a hybrid quantum model might handle simple logical relationships.



Practical tips: When experimenting, it is often easiest to start with fully visible QBMs (no hidden units). In that case, the model distribution is $p(v) \propto \bra{v}e^{-\beta H}\ket{v}$ where $v$ runs over visible states. Gradient descent on this model can be done classically for very small sizes by explicitly computing $e^{-\beta H}$ and its gradients. For larger sizes, one must rely on quantum sampling or approximations.



Example Problem: Try training a 2-qubit QBM (4-dimensional state space) to match a given target distribution, e.g.\ $p(00)=0.5, p(11)=0.5$ and $p(01)=p(10)=0$. This requires choosing a Hamiltonian $H = a,\sigma^z_1\sigma^z_2 + b,(\sigma^z_1+\sigma^z_2)$ (and $\Gamma$ fields). Intuitively, $a$ should be negative (to favor $00,11$) and $b$ controls bias. Use gradient descent to find $a,b$ that achieve the distribution at some $\beta$.





References





Montúfar, G. (2018). Restricted Boltzmann Machines: Introduction and Review. Journal of Machine Learning Research 23(36), 1-108 .
Amin, M. H., Andriyash, E., Rolfe, J., Kulchytskyy, B., & Melko, R. (2018). Quantum Boltzmann Machine. Physical Review X, 8(2), 021050 .
Askarzadeh, A., Cong, I., & Sarma, S. (2024). On the sample complexity of quantum Boltzmann machine learning. Communications Physics 7, 24 .
Wiebe, N., & Wossnig, L. (2019). Generative training of quantum Boltzmann machines with hidden units. arXiv:1905.09902 .
Tan, A.T.V., Bahr, K., Plewnia, R., Grimsley, H.R., & Cerezo, M. (2023). Training quantum Boltzmann machines with coresets. arXiv:2307.14459 .
Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural Computation 14, 1771–1800.
Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). A Fast Learning Algorithm for Deep Belief Nets. Neural Computation 18(7), 1527–1554.






Index





Boltzmann machine (BM): Pairwise energy-based model; probabilistic graphical model (Ch. 2).
Partition function: Normalization constant $Z = \sum_x e^{-E(x)}$ in BM probability (Ch. 2).
Restricted Boltzmann machine (RBM): Bipartite BM with visible–hidden layers only (Ch. 2).
Quantum Boltzmann machine (QBM): Quantum neural network with qubit-based energy (Ch. 3, 4).
Restricted QBM (RQBM): Bipartite QBM with visible and hidden qubits (Ch. 4).
Gibbs state: Thermal state $\rho=e^{-\beta H}/\Tr(e^{-\beta H})$ in a quantum model (Ch. 3).
Relative entropy: Quantum KL divergence used as training objective (Ch. 4).
Contrastive Divergence (CD): Approximate BM training algorithm (Ch. 2, 6).
PennyLane: Quantum software library for circuit simulations (Ch. 5).
Hamiltonian: Quantum operator representing energy (Ch. 3, 5).
Visibility (clamping): Fixing visible qubits to data values during training (Ch. 3–5).~
o Discussion and work on project 2
!eblock





Table of Contents





References
Index






Classical Boltzmann Machines and RBMs Review





A Boltzmann machine (BM) is an undirected probabilistic graphical model of binary units (neurons) with pairwise interactions . Each unit $i$ has a binary state $x_i\in{0,1}$ (or $\pm1$) and an associated bias $b_i$, and each pair of units $(i,j)$ may have a symmetric interaction weight $w_{ij}$. The energy of a state $x=(x_1,\dots,x_N)$ is defined as

$$

E(x; b,w) = -\sum_{i<j} w_{ij},x_i x_j - \sum_{i} b_i,x_i,,

$$

where we adopt the physics convention (hence the minus signs) so that lower energy states are more probable.  The BM defines a Gibbs (Boltzmann) distribution over states:

$$

p(x; b,w) ;=; \frac{1}{Z(b,w)} \exp(-E(x; b,w)),

\qquad Z(b,w)=\sum_{x\in{0,1}^N} \exp(-E(x; b,w)),,

$$

where $Z$ is the partition function (normalization) .  In other words, $p(x)\propto e^{-E(x)}$ and $Z$ sums $e^{-E}$ over all $2^N$ configurations .  A BM is a general energy-based model with hidden (unobserved) and visible (observed) units. In learning, the goal is to adjust $(b,w)$ so that the model distribution matches the empirical data distribution.



A Restricted Boltzmann Machine (RBM) is a special case of BM with a bipartite structure .  Units are divided into a visible layer $v\in{0,1}^n$ (observed data) and a hidden layer $h\in{0,1}^m$, and no connections exist among units within the same layer. This restriction simplifies many calculations. The energy of an RBM state is

$$

E(v,h; W,b,c) = -\sum_{i,j} W_{ij},h_j v_i - \sum_i b_i v_i - \sum_j c_j h_j,

$$

where $W_{ij}$ couples visible unit $i$ to hidden $j$, $b_i$ are visible biases and $c_j$ hidden biases. The joint probability is

$$

p(v,h) = \frac{1}{Z} e^{-E(v,h)},

$$

and the marginal probability over visible states is

$$

p(v) = \sum_{h} p(v,h) = \frac{1}{Z} \sum_{h\in{0,1}^m} \exp\bigl(\sum_{i,j} W_{ij},h_j v_i + \sum_i b_i v_i + \sum_j c_j h_j\bigr)!.

$$

Equivalently, one can show :

$$

p(v) = \frac{1}{Z} \exp\Bigl(\sum_i b_i v_i\Bigr)\prod_{j=1}^m \bigl(1 + e^{c_j + \sum_i W_{ij}v_i}\bigr).

$$

Figure below illustrates an RBM network architecture.



Figure: Restricted Boltzmann Machine (RBM). Visible units $v_i$ (green) are connected to hidden units $h_j$ (blue), with weights $W_{ij}$. No visible-visible or hidden-hidden edges exist, enabling conditional independence and efficient sampling .



Learning in a BM/RBM: Training adjusts weights so that the model reproduces data statistics. One minimizes the negative log-likelihood $-\sum_{data}\ln p(v)$ of the data. The gradient of this loss w.r.t.\ a parameter (e.g.\ a weight or bias) involves the difference of correlations between the data and the model. For example, for a general BM the update rule (stochastic gradient descent) is :

$$

\delta W_{ij} ;\propto; \langle x_i x_j\rangle_{\text{data}} ;-; \langle x_i x_j\rangle_{\text{model}}!, \qquad

\delta b_{i} ;\propto; \langle x_i \rangle_{\text{data}} ;-; \langle x_i \rangle_{\text{model}}!,

$$

where $\langle\cdot\rangle_{\text{data}}$ are averages over data samples and $\langle\cdot\rangle_{\text{model}}$ are expectations under the model distribution. Computing $\langle\cdot\rangle_{\text{model}}$ requires (in general) summing over all states or performing Gibbs sampling, which can be expensive. In practice for RBMs one often uses Contrastive Divergence (CD) or similar approximations to efficiently approximate these expectations.



RBM Sampling and CD: In an RBM, given visible $v$, the hidden units are conditionally independent:

$$

p(h_j=1\mid v) = \sigma!\bigl(c_j + \sum_i W_{ij} v_i\bigr)!,

$$

where $\sigma(x) = 1/(1+e^{-x})$ is the logistic function. Similarly, given hidden $h$, the visibles are independent. This bipartite structure allows one Gibbs step (sampling $h$ from $p(h|v)$, then $v$ from $p(v|h)$) efficiently. Contrastive Divergence training uses just a few Gibbs steps from the data to approximate the model averages .



Key properties: Classical Boltzmann machines form an exponential family of probability distributions. They can approximate any distribution arbitrarily well if sufficiently large (universal approximator), but training is generally hard (NP-hard) due to the partition function $Z$. RBMs, due to their structure, are easier to train and have been widely used in unsupervised learning (e.g.\ as building blocks of deep belief networks ).



Exercises (Chapter 2): (1) Compute partition function: For a 2-bit fully connected BM with weights $w_{12}=0.5$, biases $b_1=b_2=0$, list the energy of each state and compute $Z$ and the probability of state $(1,1)$. (2) RBM conditional probabilities: Write the expressions for $p(h_j=1\mid v)$ and $p(v_i=1\mid h)$ in terms of the RBM parameters. (3) Contrastive Divergence: Outline the CD-1 weight update rule in words and explain why it is an approximation to maximum likelihood learning.





Introduction to Quantum Boltzmann Machines (QBMs)





A Quantum Boltzmann Machine (QBM) extends a classical BM by replacing each binary unit with a qubit and generalizing the energy to a quantum Hamiltonian .  Concretely, consider a system of $N$ qubits. A convenient choice is the transverse-field Ising model (TFIM) Hamiltonian:

$$

H = -\sum_{a=1}^N \Gamma_a,\sigma^x_a ;-;\sum_{a=1}^N b_a,\sigma^z_a ;-;\sum_{a<b} u_{ab},\sigma^z_a \sigma^z_b,

$$

where $\sigma_a^x,\sigma_a^z$ are the Pauli matrices on qubit $a$, $\Gamma_a$ is a “transverse” field, $b_a$ are local fields (biases), and $u_{ab}$ are interaction strengths .  Here each qubit can be interpreted analogously to a classical unit, but the $\sigma^x$ term induces quantum superposition.  One can also include $\sigma^y$ or more complex terms, but we focus on this common ansatz.



The quantum model assigns a density matrix (mixed state) to the system via the Gibbs (thermal) state at some inverse temperature $\beta=1/T$:

$$

\rho ;=; \frac{e^{-\beta H}}{\Tr(e^{-\beta H})},.

$$

Measuring $\rho$ in the computational ($\sigma^z$) basis yields a probability distribution over bit-strings $z\in{0,1}^N$:

$$

p(z) = \bra{z} \rho \ket{z} ;=; \frac{\bra{z} e^{-\beta H} \ket{z}}{\Tr(e^{-\beta H})}.

$$

This model is a quantum generalization of the Boltzmann distribution: if $H$ were diagonal in the $z$-basis (i.e.\ $\Gamma_a=0$), then $\bra{z} e^{-\beta H} \ket{z} = e^{-\beta E(z)}$ and one recovers a classical BM. But with nonzero $\Gamma$, $H$ has off-diagonal terms, and $\rho$ generally has entanglement and non-commuting contributions.



One advantage of QBMs is expressivity: quantum Hamiltonians can capture correlations that classical ones cannot. For instance, the non-commuting $\sigma^x$ terms allow exploring multiple states in superposition. Studies suggest that even when modeling classical data, QBMs can outperform classical BMs in capturing complex patterns .  Askarzadeh et al. note that a QBM can include “more general non-commuting terms,” making it strictly more expressive .  In practice, one often treats the visible data as classical: one may fix (clamp) the “visible” qubits to data values and only allow quantum sampling over hidden qubits, similar to RBMs. But fundamentally, a QBM defines a quantum Boltzmann distribution as above .



In summary, a QBM is a physics-inspired generative model: its parameters $(\Gamma, b, u)$ define a Hamiltonian, and sampling is done by preparing the Gibbs state of that Hamiltonian. By training these parameters, the QBM is intended to learn to generate data similar to the training set.  As with classical BMs, one can consider visible and hidden qubits: visible qubits may be initialized to classical data (e.g.\ via computational-basis initialization), while hidden qubits are free to evolve quantumly.  The model is trained so that measurements on the visible qubits (after tracing out or measuring hidden qubits) reproduce the data distribution .



Why quantum?  Intuitively, using a quantum system can allow more compact representations.  For example, Gurvits and others have shown that quantum statistics can represent certain probability distributions more efficiently than classical spins.  Moreover, upcoming quantum hardware (e.g.\ quantum annealers) may naturally implement sampling from such quantum Boltzmann distributions.  While the ultimate advantage is still under research, the idea is that QBMs could be components of future quantum machine learning systems, possibly providing speedups or capturing quantum correlations in quantum data .



Representative fact:  Tan et al. (2023) remark that a QBM is “a physically motivated quantum neural network” for generative/discriminative tasks . Similarly, Amin et al. (2018) introduce a QBM learning algorithm and demonstrate via simulations that “the machine exploits its quantum nature to mimic data sets” .



Exercises (Chapter 3): (1) QBM Hamiltonian: Write the Hamiltonian $H$ for a QBM with two visible qubits and one hidden qubit, using bias parameters $(b_1,b_2)$ for visibles, $(c_1)$ for hidden, weights $W_{1},W_{2}$ between visible 1 and hidden, etc., plus transverse fields $\Gamma_i$. (2) Quantum vs Classical: In the limit $\Gamma_a\to 0$, show that the QBM reduces to a classical BM. What role does $\Gamma$ play? (3) Density matrix: Explain why the thermal state $\rho=e^{-\beta H}/Z$ is a density matrix and how measuring in the computational basis yields a probability distribution over bit-strings.





Restricted Quantum Boltzmann Machines (RQBM)





As in the classical case, one can restrict a QBM’s connectivity to simplify training. A Restricted QBM (RQBM) has a bipartite graph structure: visible qubits connect only to hidden qubits, with no interactions among visibles or among hiddens.  The Hamiltonian then has couplings $u_{v,h}$ only between a visible qubit $v$ and a hidden qubit $h$.  Formally, for $n$ visible and $m$ hidden qubits, one can write:

$$

H = -\sum_{i=1}^n b_i,\sigma^z_{v_i} - \sum_{j=1}^m c_j,\sigma^z_{h_j} - \sum_{i,j} W_{ij},\sigma^z_{v_i}\sigma^z_{h_j} ;-; \sum_{j=1}^m \Gamma_j,\sigma^x_{h_j},

$$

where we may allow transverse fields only on the hidden qubits (visible qubits are clamped to data and treated classically). This is a quantum analog of the classical RBM: visible units (qubits) have no $\sigma^z$–$\sigma^z$ interactions among themselves, only with hidden qubits.



Such an architecture can make training easier.  Wiebe and Wossnig (2019) specifically consider a class of RQBMs where the hidden-qubit Hamiltonians commute, allowing certain variational training techniques . They point out that classical pre-training or other bounds can be used in the restricted case.  In reinforcement learning applications, Amin et al. have applied an RQBM to learn policies by clamping visible inputs (state-action pairs) and letting hidden qubits represent the “value” function .



In general, the RQBM still defines a quantum Boltzmann distribution: the Gibbs state of the above Hamiltonian yields

$$

\rho = \frac{e^{-\beta H}}{\Tr e^{-\beta H}}, \qquad

P(v) = \Tr_{h}\bigl[\bra{v} \rho \ket{v}\bigr],

$$

where $\Tr_h$ denotes tracing over hidden qubits. Training then adjusts ${W,b,c,\Gamma}$ so that $P(v)$ matches the data (where $v$ runs over visible bit-strings). By enforcing the bipartite connectivity, one hopes that sampling and gradient estimates can be done more efficiently, akin to classical RBM training.



Analogy: The RQBM retains the spirit of a classical RBM but with quantum dynamics on the hidden layer. One can view it as a hybrid: the visible qubits are often “classical” (fixed in the computational basis for each data point), while hidden qubits evolve under a transverse-field Ising Hamiltonian. This structure has been used both theoretically and experimentally to realize QBMs on hardware like quantum annealers .



Exercises (Chapter 4): (1) Graph structure: Draw an RQBM with 2 visible and 2 hidden qubits, labeling the Hamiltonian terms (biases on all qubits and weights only between visible–hidden pairs). (2) Commuting hidden Hamiltonians: Why might one require hidden-qubit Hamiltonians to commute for certain training algorithms? Discuss the case where $\Gamma$ terms are equal on all hidden qubits. (3) Connection to quantum annealing: Explain how an RQBM Hamiltonian is similar to the Hamiltonians used in quantum annealing devices (e.g.\ D-Wave), and why annealers naturally sample from (or approximate) such quantum Boltzmann distributions.





Training Quantum Boltzmann Machines





Training a QBM means optimizing its parameters so that the model distribution $\rho$ fits the data distribution. A natural objective is the quantum relative entropy (quantum Kullback–Leibler divergence) between the data ensemble and the model’s Gibbs state. Equivalently, one minimizes the negative log-likelihood of the data under the QBM.  The gradient of this loss with respect to a parameter $\theta$ typically involves terms like $\partial_\theta \Tr(\rho \ln \rho_D)$, which translate to differences of expectation values. For a QBM with Hamiltonian $H(\theta)$, it can be shown that the gradient of the log-likelihood leads to updates of the form :

$$

\delta \theta ;\propto; \langle O_\theta \rangle_{\text{data}} ;-; \langle O_\theta \rangle_{\text{model}},

$$

where $O_\theta = \partial_\theta H$ is the operator conjugate to $\theta$, and the expectations are taken over the data-clamped and model Gibbs states respectively. For example, for a weight $W_{ij}$ the gradient involves $\langle \sigma^z_{v_i}\sigma^z_{h_j}\rangle$, and for a bias $b_i$ it involves $\langle\sigma^z_{v_i}\rangle$.



However, unlike the classical case, computing the model expectations $\langle O_\theta\rangle_{\text{model}}=\Tr(\rho,O_\theta)$ requires sampling from a quantum thermal state $\rho=e^{-\beta H}/Z$.  Preparing such a Gibbs state on a quantum computer is generally hard (in fact NP-hard in worst case) .  As Tan et al. note, training requires many such samples since each gradient estimate demands potentially exponential resources .



Thus, training QBM is challenging. Several approaches exist:



Quantum variational training: One can use a variational quantum circuit to approximate the Gibbs state. For example, Quantum Imaginary Time Evolution (QITE) or a layered ansatz can approximate $e^{-\beta H}\ket{+}^{\otimes n}$. The parameters of this circuit are then optimized so that its output matches data statistics.  PennyLane and other frameworks can compute gradients via parameter-shift rules for such circuits. (This approach requires a fault-tolerant device or clever heuristics.)
Bounds and approximations: Amin et al. (2018) circumvented the non-commutativity issue by introducing a bound on quantum probabilities, allowing approximate sampling . In their method, they derive a lower bound to the log-likelihood that can be sampled efficiently. They demonstrate training with and without this bound, using exact diagonalization on small systems.  The essence is to replace $e^{-\beta H}$ with an operator that is easier to sample from, controlling the error.
Restricted/commuting methods: Wiebe & Wossnig (2019) developed methods for Restricted QBMs with commuting hidden Hamiltonians. By ensuring that the hidden-qubit Hamiltonian terms commute, one can derive a variational upper bound on the quantum relative entropy and optimize it classically .  In this restricted case, the problem simplifies, and gradient evaluation can be done using linear combinations of unitaries or by sampling from simpler Hamiltonians.
Hybrid annealing: One proposal is to use quantum annealers themselves as Boltzmann samplers. If one encodes the QBM Hamiltonian (with quantum fluctuations) onto a quantum annealer, then running it at finite temperature might (approximately) sample from the needed quantum Gibbs distribution. Some works explore training annealer qubits to mimic a QBM through reverse annealing or similar techniques .
Coreset and data reduction: Recently, Tan et al. (2023) suggested using coresets (small representative data subsets) to reduce training cost . The idea is to compress the training set into a weighted coreset so that the QBM only needs to match a small number of effective data points, reducing the number of Gibbs-state preparations needed.




Overall, training a QBM often involves computing gradients of the form

$$

\partial_\theta \bigl[-\log \Tr(e^{-\beta H})\bigr] ;=; \langle \partial_\theta H \rangle_{\text{model}}!,

$$

which is then set against empirical averages.  In practice, one may replace exact gradients with approximate or stochastic estimates.  Note that when $\Gamma=0$, the model is classical and one recovers standard BM gradients .



Contrast with classical training:  For classical RBMs, the Contrastive Divergence (CD) algorithm provides an efficient approximate gradient by a short Gibbs chain . A quantum analog of CD is less straightforward, due to the need to sample a quantum state. Some proposals use one round of (partial) quantum annealing as the “negative phase” to get an estimate of model correlations. These methods are still experimental.



Exercise: Show that minimizing the quantum relative entropy $S(\rho_D|\rho_\theta)$ (where $\rho_D$ is the data density matrix and $\rho_\theta=e^{-\beta H(\theta)}/Z$) is equivalent to maximizing the log-likelihood of the data under the model. Derive the gradient $\partial_\theta S$ to confirm it yields terms of the form $\langle \partial_\theta H \rangle_{\text{data}} - \langle \partial_\theta H \rangle_{\text{model}}$. 





Practical Implementation with PennyLane





To illustrate QBM concepts, we can use PennyLane – a Python library for quantum machine learning – to implement simple generative models. Below we sketch a toy example: a 2-qubit Quantum Circuit Born Machine (QCBM), which generates a probability distribution via a parameterized quantum circuit. While a QCBM is not exactly a QBM (it has no thermal state), it serves to show how to code a quantum generative model.

import pennylane as qml
from pennylane import numpy as np

# Use a 2-qubit simulator
dev = qml.device('default.qubit', wires=2)

# Define a simple 2-qubit QCBM circuit
@qml.qnode(dev)
def circuit(params):
    # params = 4 angles
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[2], wires=0)
    qml.RY(params[3], wires=1)
    return qml.probs(wires=[0,1])  # return probabilities of |00>,|01>,|10>,|11>

# Initialize random parameters
params = np.random.randn(4, requires_grad=True)

# Get the output distribution from the circuit
probs = circuit(params)
print("Probabilities:", probs)
In this code, we create a 2-qubit variational circuit and return the probabilities of each basis state. One could train params to match a target distribution by defining a cost (e.g. KL divergence) between probs and the data distribution, and using PennyLane’s automatic differentiation to update params.  Though not a true QBM (no Hamiltonian or Gibbs state), this demonstrates how one might build and train a small quantum generative model in PennyLane.



To make it more QBM-like, one could encode the classical visible units as fixed inputs. For example, to compute the energy of a QBM Hamiltonian for a given visible bitstring, one might do:

# Example: compute expectation of a simple Hamiltonian for a given state |v>
H = 1.5 * qml.PauliZ(0) + 0.7 * qml.PauliZ(1) + 0.9 * qml.PauliZ(0)@qml.PauliZ(1)

@qml.qnode(dev)
def energy_of_state():
    # Prepare visible qubits in state |v> = |01>, say
    qml.PauliX(wires=1)  # flips qubit 1 to |1>
    # No hidden qubits in this simple example
    # Return expectation of H in this state
    return qml.expval(H)

print("Energy of |01> state:", energy_of_state())
Here, we used a 2-qubit Hamiltonian $H=1.5\sigma_z^0+0.7\sigma_z^1+0.9\sigma_z^0\sigma_z^1$ and prepared the state $|v>=|01>$. The expval(H) call returns $\langle 01|H|01\rangle = -1.5 + 0.7 - 0.9 = -1.7$ (up to sign conventions). This shows how PennyLane can compute energies of basis states, which is a key step in evaluating the QBM energy and probability of data states.



In practice, training a QBM in PennyLane would involve preparing quantum states corresponding to the Gibbs distribution. One could use functions like qml.qaoa.cost.Hamiltonian or custom circuits to approximate $e^{-\beta H}$, and then use PennyLane’s optimizers. Also, PennyLane can interface with PyTorch or TensorFlow, enabling hybrid optimization of quantum circuits with classical parameters (e.g. the Hamiltonian weights).



Summary of steps in PennyLane implementation:



Define a quantum device (qml.device) with enough wires for visibles+hidden.
Construct a parametric quantum circuit (ansatz) that depends on trainable parameters (e.g. angles in rotations and entangling gates).
If modeling an RQBM, one might clamp visible qubits (preparing them according to training data) and apply gates only to hidden qubits.
Measure relevant observables: one can return probabilities (probs) or expectation values (expval) of Pauli operators, depending on the chosen loss function.
Define a cost function, such as the negative log-likelihood or a distance between output and target distribution.
Use an optimizer (e.g. gradient descent, Adam) with PennyLane’s gradient calculations to update parameters.




Below is a skeleton illustrating training a QCBM on a simple 2-point dataset. This is pseudocode for illustration only:

data = np.array([[0,1], [1,0]])  # two data points (|01> and |10>)

def cost(params):
    loss = 0
    for v in data:
        # clamp visible by preparing state |v>
        @qml.qnode(dev)
        def circuit_clamped():
            if v[0]==1: qml.PauliX(wires=0)
            if v[1]==1: qml.PauliX(wires=1)
            # apply layers on all qubits (visible hidden together in this toy example)
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return qml.probs(wires=[0,1])
        prob = circuit_clamped()
        # compute negative log prob of data state (this is a toy loss)
        idx = v[0]*2 + v[1]  # index of bitstring in probs
        loss += -np.log(prob[idx] + 1e-6)
    return loss / len(data)

opt = qml.GradientDescentOptimizer(stepsize=0.1)
for step in range(50):
    params, curr_cost = opt.step_and_cost(cost, params)
    if step%10==0:
        print(f"Step {step}: cost = {curr_cost:.3f}")
In this toy loop, we optimize the circuit so that the probability of the given data points is high. In a true QBM, the cost would be derived from the log-likelihood of all data under the quantum thermal model.



Exercises (Chapter 5): (1) PennyLane circuit: Modify the above QCBM code to use 3 qubits and train to model a three-bit data distribution. (2) Hamiltonian expectation: Using PennyLane, compute the expectation value of $H = \sigma^x\otimes\sigma^x + \sigma^z\otimes\sigma^z$ on the Bell state $(|00>+|11>)/\sqrt{2}$. (3) Parameter shift rule: Explain how PennyLane can compute gradients of quantum circuit parameters via the parameter-shift rule, and why this is useful for QBM training.





Applications and Examples





Although QBMs are still mainly a research topic, we briefly mention some toy applications and encourage experimentation with small datasets.  Classical RBMs have been used for simple generative tasks like the Bars-and-Stripes dataset (binary images) or the MNIST digit dataset.  QBMs could similarly be tested on these tasks to see if they offer any advantage. For example, one can train a small QBM to generate $4\times4$ binary patterns.  Tan et al. (2023) used QBMs on the Bars-and-Stripes task (4×4 images) as a benchmark .



In reinforcement learning, QBM-like models have been used to learn policies: one treats (state,action) pairs as inputs and uses the QBM to approximate a Q-function .  An RQBM with input neurons (for state), output neurons (for action), and hidden qubits can learn the value of each action by minimizing a “free energy” (analogous to Q-value).  This demonstrates how QBMs could serve as function approximators.



Small dataset example: Consider a 4-bit dataset consisting of the two strings 0000 and 1111. A classical RBM with sufficient hidden units can exactly model this. A QBM with 4 visible qubits and some hidden qubits could also learn it. One could set up the QBM Hamiltonian and run a variational algorithm so that $|0000>$ and $|1111>$ have highest probability.  Training could be done by maximizing $p(0000)+p(1111)$. This is an example of a bimodal distribution that a BM learns.



Another toy example is the XOR problem: classify inputs $[0,0],[0,1],[1,0],[1,1]$ with target labels given by XOR. A small QBM or RQBM could be used as a generative model over the four inputs and supervised labels, similar to a discriminative model. Although this is artificial, it illustrates how a hybrid quantum model might handle simple logical relationships.



Practical tips: When experimenting, it is often easiest to start with fully visible QBMs (no hidden units). In that case, the model distribution is $p(v) \propto \bra{v}e^{-\beta H}\ket{v}$ where $v$ runs over visible states. Gradient descent on this model can be done classically for very small sizes by explicitly computing $e^{-\beta H}$ and its gradients. For larger sizes, one must rely on quantum sampling or approximations.



Example Problem: Try training a 2-qubit QBM (4-dimensional state space) to match a given target distribution, e.g.\ $p(00)=0.5, p(11)=0.5$ and $p(01)=p(10)=0$. This requires choosing a Hamiltonian $H = a,\sigma^z_1\sigma^z_2 + b,(\sigma^z_1+\sigma^z_2)$ (and $\Gamma$ fields). Intuitively, $a$ should be negative (to favor $00,11$) and $b$ controls bias. Use gradient descent to find $a,b$ that achieve the distribution at some $\beta$.





References





Montúfar, G. (2018). Restricted Boltzmann Machines: Introduction and Review. Journal of Machine Learning Research 23(36), 1-108 .
Amin, M. H., Andriyash, E., Rolfe, J., Kulchytskyy, B., & Melko, R. (2018). Quantum Boltzmann Machine. Physical Review X, 8(2), 021050 .
Askarzadeh, A., Cong, I., & Sarma, S. (2024). On the sample complexity of quantum Boltzmann machine learning. Communications Physics 7, 24 .
Wiebe, N., & Wossnig, L. (2019). Generative training of quantum Boltzmann machines with hidden units. arXiv:1905.09902 .
Tan, A.T.V., Bahr, K., Plewnia, R., Grimsley, H.R., & Cerezo, M. (2023). Training quantum Boltzmann machines with coresets. arXiv:2307.14459 .
Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural Computation 14, 1771–1800.
Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). A Fast Learning Algorithm for Deep Belief Nets. Neural Computation 18(7), 1527–1554.






Index





Boltzmann machine (BM): Pairwise energy-based model; probabilistic graphical model (Ch. 2).
Partition function: Normalization constant $Z = \sum_x e^{-E(x)}$ in BM probability (Ch. 2).
Restricted Boltzmann machine (RBM): Bipartite BM with visible–hidden layers only (Ch. 2).
Quantum Boltzmann machine (QBM): Quantum neural network with qubit-based energy (Ch. 3, 4).
Restricted QBM (RQBM): Bipartite QBM with visible and hidden qubits (Ch. 4).
Gibbs state: Thermal state $\rho=e^{-\beta H}/\Tr(e^{-\beta H})$ in a quantum model (Ch. 3).
Relative entropy: Quantum KL divergence used as training objective (Ch. 4).
Contrastive Divergence (CD): Approximate BM training algorithm (Ch. 2, 6).
PennyLane: Quantum software library for circuit simulations (Ch. 5).
Hamiltonian: Quantum operator representing energy (Ch. 3, 5).
Visibility (clamping): Fixing visible qubits to data values during training (Ch. 3–5).
