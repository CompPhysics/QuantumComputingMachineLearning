from  matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
dim = 2
Hamiltonian = np.zeros((dim,dim))
e0 = 0.0
e1 = 4.0
Xnondiag = 0.20
Xdiag = 3.0
Eigenvalue = np.zeros(dim)
# setting up the Hamiltonian
Hamiltonian[0,0] = Xdiag+e0
Hamiltonian[0,1] = Xnondiag
Hamiltonian[1,0] = Hamiltonian[0,1]
Hamiltonian[1,1] = e1-Xdiag
# diagonalize and obtain eigenvalues, not necessarily sorted
EigValues, EigVectors = np.linalg.eig(Hamiltonian)
permute = EigValues.argsort()
EigValues = EigValues[permute]
# print only the lowest eigenvalue
print(EigValues[0])

# Now rewrite it in terms of the identity matrix and the Pauli matrix X and Z
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
# identity matrix
I = np.array([[1,0],[0,1]])

epsilon = (e0+e1)*0.5; omega = (e0-e1)*0.5
c = 0.0; omega_z=Xdiag; omega_x = Xnondiag
Hamiltonian = (epsilon+c)*I+(omega_z+omega)*Z+omega_x*X
EigValues, EigVectors = np.linalg.eig(Hamiltonian)
permute = EigValues.argsort()
EigValues = EigValues[permute]
# print only the lowest eigenvalue
print(EigValues[0])

# define the rotation matrices

def Rx(theta):
    return np.cos(theta*0.5)*I-1j*np.sin(theta*0.5)*X
def Ry(phi):
    return np.cos(phi*0.5)*I-1j*np.sin(phi*0.5)*Y

#define basis states
basis0 = np.array([1,0])
basis1 = np.array([0,1])

# Computing the expectation value of the energy 
def Energy(theta,phi):
    Basis = Ry(phi) @ Rx(theta) @ basis0
    energy = Basis.conj().T @ Hamiltonian @ Basis
    return energy


# Set up iteration using gradient descent method
eta = 0.1
Niterations = 100
# Random angles using uniform distribution
theta = 2*np.pi*np.random.rand()
phi = 2*np.pi*np.random.rand()
pi2 = 0.5*np.pi
for iter in range(Niterations):
    thetagradient = 0.5*(Energy(theta+pi2,phi)-Energy(theta-pi2,phi))
    phigradient = 0.5*(Energy(theta,phi+pi2)-Energy(theta,phi-pi2))
    theta -= eta*thetagradient
    phi -= eta*phigradient
print(Energy(theta,phi))
