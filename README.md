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
- Hidary, Quantum Computing: An Applied Approach, see https://link.springer.com/book/10.1007/978-3-030-23922-0
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
  - Video of lecture at https://youtu.be/J5lK-fTcTYY
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/HandWrittenNotes/2026/Lectureweek1.pdf
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week1
  - _Reading recommendation_: Scherer chapter 2 and/or Hidary chapter 12

## January 26 - January 30, 2026. Composite Systems and Tensor Products
  - Spectral decomposition and measurements
  - Density matrices
  - Entanglement, pure and mixed states
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week2  
  - _Reading recommendation_: Scherer chapter 2 and sections 3.1-3.3. Hundt, Quantum Computing for Programmers, chapter 2.1-2.5. Hundt's text is relevant for the programming part where we build from scratch the ingredients we will need.
  - Video of lecture at https://youtu.be/KgMw2lC-oJY
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek2.pdf

## February 2-6, 2026. Density matrices and Measurements
  - Density matrices, entanglement and entropies
  - Video of lecture at https://youtu.be/xkbXx6XIlvU
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek3.pdf 
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week3  

## February 9-13, 2026. Entanglement and entropies
  - Reminder from last week on entanglement, density matrices and entropies
  - One-qubit and two-qubit gates, background and realizations
  - Simple Hamiltonian systems and getting started with the first project
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week4  
  - _Reading recommendation_: For the discussion of one-qubit, two-qubit and other gates, sections 2.6-2.11 and 3.1-3.4 of Hundt's book Quantum Computing for Programmers, contain most of the relevant information.
  - Video of lecture at https://youtu.be/4Ew5UNHnsdM
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek4.pdf
  
## February 16-20, 2026. Getting started with the VQE algorithm
  - Quantum gates and operations and simple quantum algorithms
  - Discussion of the VQE algorithm and discussions of project 1
  - Simple one-qubit and two-qubit Hamiltonians
  - Video of lecture at https://youtu.be/f9AfWyGWbCI
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lecturesweek5.pdf
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week5. See in particular the additional jupyter-notebooks for the one- and two-qubit cases. For those of you who wish to test IBM's quantum computers with Qiskit, there are similar notebooks.
  - _Reading recommendation_: For the discussion of one-qubit, two-qubit and other gates, sections 2.6-2.11, 3.1-3.4 and 6.11.1-.6.11.3 of Hundt's book Quantum Computing for Programmers, contain most of the relevant information.


## February 23-27, 2026. Implementing the VQE with measurements and evaluation of gradients
  - VQE and adaptive VQE, Variational Quantum Eigensolver and discussion of codes
  - Simulations of  of Hamiltonians, focus on the one- and two-qubit Hamiltonians
  - Start discussions of Lipkin model
  - Video of lecture at https://youtu.be/MVLbBcTPwqg
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek6.pdf
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week6



##  March 2-6, 2026. VQE for two-qubit systems and the Lipkin model
  - Implementing the VQE algorithm for the two-qubit and Lipkin-model Hamiltonians.
  - Video of lecture at https://youtu.be/g-hKlUYxfcw
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek7.pdf
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week7


##  March 9-13, 2026. Solving quantum mechanical problems
  - Lipkin model and VQE
  - Jordan-Wigner transformation and other Hamiltonians as examples
  - Start discussion of Quantum Fourier Transforms
  - Lab/exercise session: work on project 1
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week8
  - Video of lecture at https://youtu.be/C8vxBY-AmD8
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek8.pdf


## March 16-20, 2026. Discussions of project 1 and start discussion of Quantum Fourier transofrms
  - Quantum Fourier transforms
  - Lab/exercise session: Discussion of project 1 and work on finalizing project 1
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week9
  - Video of lecture at https://youtu.be/qQw4zme7LyI
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek9.pdf


## March 23-27, 2026
  - Quantum Fourier Transforms, algorithm and implementation
  - Quantum phase estimation (QPE) algorithm
  - Setting up circuits for QFTs and the QPE  and discussion of codes
  - Teaching material in different formats at https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/pub/week10
  - Video of lecture at https://youtu.be/N9AXHW72qwI
  - Whiteboard notes at https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/HandWrittenNotes/2026/Lectureweek10.pdf

## March 30 - April 3, 2026, Public holiday in Norway no classes

## April 6-10, 2026
  - The QAOA algorithm
  - The HHL algorithm for solving linear algebra problems

## April 13-17, 2026
  - Basics of quantum machine learning (QML) 
  - Support vector machines

##  Note that the topics for the coming weeks may change. The focus will be on quantum machine learning, but we may add discussions of other algorithms

## April 20-24, 2026 Quantum Machine Learning
  - Quantum machine learning

## April 27-May 1, 2026 Quantum machine learning
  - QML

## May 4-8, 2026 Quantum Machine Learning
  - QML

## May 11-15, 2026
  - QML, last lecture and summary of course

## May 18-22, 2026
  - Discussion of and work on project 2
