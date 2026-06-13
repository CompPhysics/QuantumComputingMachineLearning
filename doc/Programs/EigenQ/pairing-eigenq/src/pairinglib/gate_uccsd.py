"""Gate-level UCCSD: fermionic excitations compiled to rotation+CNOT circuits."""
import numpy as np
from scipy.optimize import minimize
from ._exc import (make_single, make_double, apply_exc_circuit,
                   jw_create, jw_ann)
from .hamiltonian import build_H_full
from .uccsd import uccsd_pool

__all__ = ["make_single", "make_double", "apply_exc_circuit",
           "gate_uccsd_setup", "gate_uccsd_vqe", "gate_uccsd_state"]

def gate_uccsd_setup(k, N):
    nq = 2*k; singles, doubles = uccsd_pool(k, N)
    terms = [make_single(i, a, nq) for (i, a) in singles] \
          + [make_double(i, j, a, b, nq) for (i, j, a, b) in doubles]
    gens = []
    for (i, a) in singles:
        G = jw_create(a, nq) @ jw_ann(i, nq); gens.append(G - G.conj().T)
    for (i, j, a, b) in doubles:
        G = jw_create(a, nq) @ jw_create(b, nq) @ jw_ann(j, nq) @ jw_ann(i, nq)
        gens.append(G - G.conj().T)
    hf = sum(1 << (nq-1-j) for j in range(N))
    ncnot = sum(2*(len(P)-1) for tm in terms for (P, _) in tm)
    nrot = sum(len(tm) for tm in terms)
    return dict(nq=nq, terms=terms, gens=gens, hf=hf, P=len(terms),
                ncnot=ncnot, nrot=nrot)

def gate_uccsd_vqe(k, N, g, setup, delta=1.0):
    nq = setup['nq']; dim = 2**nq; terms = setup['terms']; gens = setup['gens']
    hf = setup['hf']; P = setup['P']; H = build_H_full(k, g, delta)
    def state(th):
        psi = np.zeros(dim, complex); psi[hf] = 1.0
        for m in range(P): psi = apply_exc_circuit(psi, th[m], terms[m], nq)
        return psi
    def energy(th): psi = state(th); return float(np.real(psi.conj() @ (H @ psi)))
    def grad(th):
        psi = np.zeros(dim, complex); psi[hf] = 1.0; st = [psi.copy()]
        for m in range(P): psi = apply_exc_circuit(psi, th[m], terms[m], nq); st.append(psi.copy())
        Hpsi = H @ st[P]; gg = np.zeros(P)
        for mu in range(P):
            v = gens[mu] @ st[mu+1]
            for m in range(mu+1, P): v = apply_exc_circuit(v, th[m], terms[m], nq)
            gg[mu] = 2.0*float(np.real(v.conj() @ Hpsi))
        return gg
    if P == 0: return energy(np.zeros(0)), 0
    res = minimize(energy, np.zeros(P), jac=grad, method='L-BFGS-B',
                   options={'ftol': 1e-13, 'gtol': 1e-11, 'maxiter': 2000})
    return res.fun, res.nit

def gate_uccsd_state(k, N, g):
    """Optimise the gate-UCCSD circuit; return (energy, full-space statevector)."""
    su = gate_uccsd_setup(k, N); nq = su['nq']; dim = 2**nq
    H = build_H_full(k, g); terms = su['terms']; gens = su['gens']
    hf = su['hf']; P = su['P']
    def state(th):
        psi = np.zeros(dim, complex); psi[hf] = 1.0
        for m in range(P): psi = apply_exc_circuit(psi, th[m], terms[m], nq)
        return psi
    def energy(th): psi = state(th); return float(np.real(psi.conj() @ (H @ psi)))
    def grad(th):
        psi = np.zeros(dim, complex); psi[hf] = 1.0; st = [psi.copy()]
        for m in range(P): psi = apply_exc_circuit(psi, th[m], terms[m], nq); st.append(psi.copy())
        Hpsi = H @ st[P]; gg = np.zeros(P)
        for mu in range(P):
            v = gens[mu] @ st[mu+1]
            for m in range(mu+1, P): v = apply_exc_circuit(v, th[m], terms[m], nq)
            gg[mu] = 2.0*float(np.real(v.conj() @ Hpsi))
        return gg
    if P == 0: return energy(np.zeros(0)), state(np.zeros(0))
    res = minimize(energy, np.zeros(P), jac=grad, method='L-BFGS-B',
                   options={'ftol': 1e-13, 'gtol': 1e-11, 'maxiter': 2000})
    return res.fun, state(res.x)
