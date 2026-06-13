"""Rodeo algorithm: stochastic spectral filtering (Qian et al., EPJA 60, 151)."""
import numpy as np
from scipy.linalg import expm

__all__ = ["rodeo_post0", "rodeo_cycle_ancilla", "rodeo_sweep",
           "rodeo_allzero_prob", "rodeo_track"]

def rodeo_post0(v, U, t, E):
    """Post-selected |0> branch of one cycle = (I + e^{iEt} U)/2 . Returns (state, prob)."""
    w = 0.5*(v + np.exp(1j*E*t)*(U @ v))
    p = float(np.vdot(w, w).real)
    return w/np.sqrt(p), p

def rodeo_cycle_ancilla(v, U, t, E):
    """Explicit ancilla circuit (H - ctrl-U - phase - H - measure|0>); same map as rodeo_post0."""
    D = len(v); J = np.zeros((D, 2), complex); J[:, 0] = v
    a0, a1 = J[:, 0], J[:, 1]; J = np.stack([(a0+a1), (a0-a1)], axis=1)/np.sqrt(2)
    J[:, 1] = U @ J[:, 1]; J[:, 1] = J[:, 1]*np.exp(1j*E*t)
    a0, a1 = J[:, 0], J[:, 1]; J = np.stack([(a0+a1), (a0-a1)], axis=1)/np.sqrt(2)
    w = J[:, 0]; p = float(np.vdot(w, w).real)
    return w/np.sqrt(p), p

def rodeo_sweep(v0, H, ts, E):
    """Apply len(ts) cycles; return (final state, cumulative acceptance array)."""
    v = v0.astype(complex).copy(); P = 1.0; Ps = []
    for t in ts:
        v, p = rodeo_post0(v, expm(-1j*H*t), t, E); P *= p; Ps.append(P)
    return v, np.array(Ps)

def rodeo_allzero_prob(v0, H, ts, E):
    """Joint all-zero acceptance probability for one time set at target E."""
    return rodeo_sweep(v0, H, ts, E)[1][-1]

def rodeo_track(v0, H, ts, E, target):
    """Return fidelity-with-`target` and cumulative acceptance after each cycle."""
    v = v0.astype(complex).copy(); P = 1.0; F, A = [], []
    for t in ts:
        v, p = rodeo_post0(v, expm(-1j*H*t), t, E); P *= p
        F.append(abs(np.vdot(target, v))**2); A.append(P)
    return np.array(F), np.array(A)
