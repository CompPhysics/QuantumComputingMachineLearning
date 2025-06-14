from qiskit import QuantumCircuit
import numpy as np

class Gates:
    """Wrapper class for basic gates"""
    
    @staticmethod
    def I(qc, qubit):
        qc.id(qubit)
    
    @staticmethod
    def X(qc, qubit):
        qc.x(qubit)
    
    @staticmethod
    def Y(qc, qubit):
        qc.y(qubit)
    
    @staticmethod
    def Z(qc, qubit):
        qc.z(qubit)
   @staticmethod
    def H(qc, qubit):
        qc.h(qubit)
    
    @staticmethod
    def S(qc, qubit):
        qc.s(qubit)
    
    @staticmethod
    def T(qc, qubit):
        qc.t(qubit)
    
    @staticmethod
    def RX(qc, qubit, theta):
        qc.rx(theta, qubit)
    
    @staticmethod
    def RY(qc, qubit, theta):
        qc.ry(theta, qubit)
    
    @staticmethod
    def RZ(qc, qubit, theta):
        qc.rz(theta, qubit)
    
    @staticmethod
    def CNOT(qc, control, target):
        qc.cx(control, target)
    
    @staticmethod
    def CZ(qc, control, target):
        qc.cz(control, target)
    
    @staticmethod
    def SWAP(qc, qubit1, qubit2):
        qc.swap(qubit1, qubit2)
