import numpy as np
from scipy.linalg import expm

# ---- statevector gate simulator, big-endian: qubit j <-> bit (n-1-j) ----
def apply_1q(psi, U, q, n):
    psi=psi.reshape((2**q,2,2**(n-1-q)))
    psi=np.einsum('ab,ibj->iaj',U,psi)
    return psi.reshape(-1)

def apply_cnot(psi, c, t, n):
    idx=np.arange(2**n)
    ctrl=((idx>>(n-1-c))&1)==1
    perm=np.where(ctrl, idx ^ (1<<(n-1-t)), idx)
    return psi[perm]

# ---- gates ----
def Ry(th):
    c,s=np.cos(th/2),np.sin(th/2); return np.array([[c,-s],[s,c]],complex)
def Rz(th):
    return np.array([[np.exp(-1j*th/2),0],[0,np.exp(1j*th/2)]],complex)
def Rx(th):
    c,s=np.cos(th/2),np.sin(th/2); return np.array([[c,-1j*s],[-1j*s,c]],complex)
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
Sdg=np.array([[1,0],[0,-1j]],complex)
S=np.array([[1,0],[0,1j]],complex)

# ---- exp(-i phi/2 * PauliString) circuit ----
# pauli: dict {qubit: 'X'/'Y'/'Z'}
def basis_change(psi,pauli,n,inverse=False):
    for q,P in pauli.items():
        if P=='Z': continue
        if P=='X': U=H
        elif P=='Y': U=(H@Sdg) if not inverse else (S@H)
        psi=apply_1q(psi,U,q,n)
    return psi

def pauli_exp(psi, phi, pauli, n):
    """Apply exp(-i*phi/2 * P) where P=tensor of pauli ops. Returns new psi."""
    qs=sorted(pauli.keys())
    psi=basis_change(psi,pauli,n,inverse=False)
    # CNOT ladder onto last qubit
    for a,b in zip(qs[:-1],qs[1:]):
        psi=apply_cnot(psi,a,b,n)
    psi=apply_1q(psi,Rz(phi),qs[-1],n)
    for a,b in zip(qs[:-1],qs[1:])[::-1] if False else list(zip(qs[:-1],qs[1:]))[::-1]:
        psi=apply_cnot(psi,a,b,n)
    psi=basis_change(psi,pauli,n,inverse=True)
    return psi

# ---- verify pauli_exp against matrix exponential ----
def pauli_matrix(pauli,n):
    Pm={'I':np.eye(2,dtype=complex),'X':np.array([[0,1],[1,0]],complex),
        'Y':np.array([[0,-1j],[1j,0]],complex),'Z':np.array([[1,0],[0,-1]],complex)}
    mats=[Pm[pauli.get(q,'I')] for q in range(n)]
    M=mats[0]
    for m in mats[1:]: M=np.kron(M,m)
    return M

if __name__=='__main__':
    rng=np.random.default_rng(0); n=4
    for trial in range(5):
        pauli={1:'Y',2:'Z',3:'X'}
        phi=rng.uniform(-2,2)
        psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n); psi/=np.linalg.norm(psi)
        out=pauli_exp(psi.copy(),phi,pauli,n)
        ref=expm(-1j*phi/2*pauli_matrix(pauli,n))@psi
        print(f"pauli_exp vs expm: max diff = {np.max(np.abs(out-ref)):.2e}")
