import numpy as np

# One-qubit gates
def I():  return np.eye(2)
def X():  return np.array([[0,1],[1,0]])
def Y():  return np.array([[0,-1j],[1j,0]])
def Z():  return np.array([[1,0],[0,-1]])
def H():  return (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
def S():  return np.array([[1,0],[0,1j]])
def T():  return np.array([[1,0],[0,np.exp(1j*np.pi/4)]])

def Rx(theta):
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])

def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2),  np.cos(theta/2)]
    ])

def Rz(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ])

# Two-qubit gates
def CNOT():
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0]
    ])

def CZ():
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,-1]
    ])

def SWAP():
    return np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ])

