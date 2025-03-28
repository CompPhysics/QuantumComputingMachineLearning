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

# define also the rotation matrices
# Define angles theta and phi
theta = 0.5*np.pi; phi = 0.2*np.pi
Rx = np.cos(theta*0.5)*I-1j*np.sin(theta*0.5)*X
Ry = np.cos(phi*0.5)*I-1j*np.sin(phi*0.5)*Y
#define basis states
basis0 = np.array([1,0])
basis1 = np.array([0,1])

NewBasis = Ry @ Rx @ basis0
# Compute the expectation value
Energy = NewBasis.conj().T @ Hamiltonian @ NewBasis
print(Energy)

# Computing the derivative of the energy and the energy 
def EnergyDerivative(x0):
    theta = x0[0]
    phi = x0[1]
    Rx = np.cos(theta*0.5)*I-1j*np.sin(theta*0.5)*X
    Ry = np.cos(phi*0.5)*I-1j*np.sin(phi*0.5)*Y
    Basis = Ry @ Rx @ basis0
    XX = -1j*0.5*X; YY = -1j*0.5*Y
    d1 = (Basis.conj().T @ Hamiltonian @ XX @ Basis)#+(Basis.conj().T @ Hamiltonian @ XX @ Basis).conj()
    EDerivative[0] = d1
    d2 = (Basis.conj().T @ Hamiltonian @ YY @ Basis)#+(Basis.conj().T @ Hamiltonian @ YY @ Basis).conj()
    EDerivative[1] = d2
    
    return EDerivative


# Computing the expectation value of the energy 
def Energy(x0):
    theta = x0[0]
    phi = x0[1]
    Rx = np.cos(theta*0.5)*I-1j*np.sin(theta*0.5)*X
    Ry = np.cos(phi*0.5)*I-1j*np.sin(phi*0.5)*Y
    Basis = Ry @ Rx @ basis0
    energy = Basis.conj().T @ Hamiltonian @ Basis
    return energy



# Set up iteration using gradient descent method
Energy = 0
EDerivative = np.zeros(2,dtype=np.complex_)
eta = 0.001
Niterations = 100
x0 = np.array([0.5*np.pi,0.2*np.pi])
for iter in range(Niterations):
    EDerivative = EnergyDerivative(x0) 
    thetagradient = EDerivative[0]
    phigradient = EDerivative[1]
    theta -= eta*thetagradient
    phi -= eta*phigradient
    x0 = np.array([theta,phi])
    print(EDerivative)
#print(Energy(x0))

"""
# Using Broydens method
res = minimize(Energy, x0, method='BFGS', jac=EnergyDerivative, options={'gtol': 1e-4,'disp': True})
print(res.x)
"""
