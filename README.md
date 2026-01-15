# Quantum Computing and Quantum Machine Learning 

The first part of the course (project 1 and till mid march) has its focus on studies of
quantum-mechanical many-particle systems using quantum computing
algorithms and quantum computers. The second part is optional and
depends on the interests and backgrounds of the participants. Two main
themes can be covered:
- Quantum machine learning algorithms, implementations and studies
- Realization and studies of entanglement in physical systems
- Advanced VQE and hamiltonian systems
### Possible  textbooks:
- Maria Schuld and Francesco Petruccione, Machine Learning with Quantum Computers, see https://link.springer.com/book/10.1007/978-3-030-83098-4
- Wolfgang Scherer, Mathematics of Quantum Computing, see https://link.springer.com/book/10.1007/978-3-030-12358-1
- Robert Hundt, Quantum Computing for Programmers, https://www.cambridge.org/core/books/quantum-computing-for-programmers/BA1C887BE4AC0D0D5653E71FFBEF61C6
- Claudio Conti, Quantum Machine Learning (Springer), https://link.springer.com/book/10.1007/978-3-031-44226-1
- Robert Loredo, Learn Quantum Computing with Python and IBM Quantum Experience, see https://github.com/PacktPublishing/Learn-Quantum-Computing-with-Python-and-IBM-Quantum-Experience
- Stefano Olivares, A Student’s Guide to Quantum Computing, see https://link.springer.com/book/10.1007/978-3-031-83361-8

### Interesting online courses and software:
- IBM's Quantum Computer Programming: Hands-On Workshop at https://quantgates.com/learn-quantum
- QuTip at https://github.com/qutip and https://qutip.org/
- QisKit at https://www.ibm.com/quantum/qiskit
- PySCF for traditional quantum mechanical methods at https://pyscf.org/user/install.html#how-to-install-pyscf. This library can be integrated with QisKit for quantum computing simulations.
- Qbraid at https://www.qbraid.com
- PennyLane at https://pennylane.ai/ (tailored to machine learning)

### Time: Each Wednesday at 1015am-12pm CET and exercise sessions 815-10am (The lecture sessions will be recorded)
-Permanent Zoom link for the whole semester is https://uio.zoom.us/my/mortenhj


## January 19-23, 2026. Overview of first week, Basic Notions of Quantum Mechanics
- Definitions, Linear Algebra reminder, Hilbert Space, Operators on Hilbert Spaces, Composite Systems
  - Definitions
  - Mathematical notation, Hilbert spaces and operators
  - Description of Quantum Systems and one-qubit systems 
  - States in Hilbert Space, pure and mixed states
  - Video of lecture to be added  https://youtu.be/
- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week1
  - _Reading recommendation_: Scherer chapter 2 

## January 26 - January 30, 2026. Composite Systems and Tensor Products
  - Spectral decomposition and measurements
  - Density matrices
  - Entanglement, pure and mixed states
v- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week2  
  - _Reading recommendation_: Scherer chapter 2 and sections 3.1-3.3. Hundt, Quantum Computing for Programmers, chapter 2.1-2.5. Hundt's text is relevant for the programming part where we build from scratch the ingredients we will need.
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesJanuary29.pdf 

## February 2-6, 2026. Density matrices and Measurements
  - Discussion of gates and project 1
  - Quantum gates and circuits
  - Developing our own codes for Bell states and comparing with qiskit
- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week3  

## February 9-13, 2026. Entanglement and entropies
  - Reminder from last week on gates and circuits
  - One-qubit and two-qubit gates, background and realizations
  - Simple Hamiltonian systems
- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week4  
  - _Reading recommendation_: For the discussion of one-qubit, two-qubit and other gates, sections 2.6-2.11 and 3.1-3.4 of Hundt's book Quantum Computing for Programmers, contain most of the relevant information.
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesFebruary12.pdf
  

## February 16-20, 2026.
  - Entanglement and Schmidt decomposition 
  - Entropy as a measurement of entanglement
  - Simple one-qubit and two-qubit Hamiltonians
- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week5  
  - _Reading recommendation_: For the discussion of one-qubit, two-qubit and other gates, sections 2.6-2.11, 3.1-3.4 and 6.11.1-.6.11.3 of Hundt's book Quantum Computing for Programmers, contain most of the relevant information.
- Video of lecture to be added  https://youtu.be/
- Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesFebruary19.pdf

## February 23-27, 2026. Quantum gates and circuits and Quantum Fourier Transform and Hamiltonians
  - Quantum gates and operations and simple quantum algorithms
  - Discussion of the VQE algorithm and discussions of project 1
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesFebruary26.pdf
- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week6

##  March 2-6, 2026. Algorithms for solving quantum mechanical problems.
  - VQE and adaptive VQE, Variational Quantum Eigensolver and discussion of codes
  - Simulations of  of Hamiltonians, focus on the one- and two-qubit Hamiltonians
  - Start discussions of Lipkin model
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesMarch5.pdf
  
- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week7


##  March 9-13, 2026. Solving quantum mechanical problems
  - Lipkin model and VQE
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week8
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesMarch12.pdf


## March 16-20, 2026. Discussions of project 1 and work on the VQE
  - Lipkin model and VQE
  - Discussion of project 1 and work on finalizing project
- Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week9

## March 23-27, 2026
  - Quantum Fourier Transforms, algorithm and implementation
  - Quantum phase estimation algorithm
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesMarch26.pdf
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week10


## March 30 - April 3, 2026, Public holiday in Norway no classes


## April 6-10, 2026
  - Discrete Fourier transforms (DFTs, reminder from last week) ) and the fast Fourier Transform (FFT)
  - Quantum Fourier transforms (QFTs), reminder from last week
  - Setting up circuits for QFTs
  - Reading recommendation Hundt, Quantum Computing for Programmers, sections 6.1-6.4 on QFT and QPE.

## April 13-17, 2026
  - Setting up circuits for QFTs
  - Quantum phase estimation algorithm (QPE)
  - Reading recommendation Hundt, Quantum Computing for Programmers, sections 6.1-6.4 on QFT and QPE.
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesApril7.pdf



## April 20-24, 2026 Quantum Machine Learning
  - Basics of quantum machine learning and discussion of support vector machines
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesApril23.pdf  

## April 27-May 1, 2026 Quantum machine learning
  - Classical Support Vector Machines, reminder from last week
  - Classical Kernels and transition to Quantum Kernels
  - Quantum Support Vector Machines
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesApril30.pdf


## May 4-8, 2026 Quantum Machine Learning
  - Quantum support vector machines, theory and code examples
  - Quantum neural networks, theory and code examples
  - Video of lecture to be added at https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesMay7.pdf

## May 11-15, 2026
  - Quantum neural networks, theory and code examples, contn from last week
  - Quantum and classical Boltzmann machines
  - Video of lecture to be added  https://youtu.be/
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/NotesMay14.pdf


## May 18-22, 2026
  - Discussion of project 2
