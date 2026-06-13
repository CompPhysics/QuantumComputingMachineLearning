"""Prolongation operator and adiabatic resolution refinement."""
import numpy as np
from scipy.linalg import expm
from .hamiltonian import build_sector, H_pairing_sparse, _bit

__all__ = ["prolong_matrix", "refine_state", "refine_NK"]

def prolong_matrix(k_low, k_high, N):
    """Basis-refinement embedding: new high-resolution levels start empty."""
    nl, sl, il = build_sector(k_low, N); nh, sh, ih = build_sector(k_high, N)
    Pm = np.zeros((len(sh), len(sl)))
    for Il in sl:
        Ih = 0
        for j in range(nl):
            if _bit(Il, j, nl): Ih |= 1 << (nh-1-j)
        Pm[ih[Ih], il[Il]] = 1.0
    return Pm

def refine_state(N, k_high, T, k_low=2, M=160, mu_buf=0.6, g=1.0):
    """Adiabatically refine the prolonged low-res ground state; return (psi, H_high)."""
    nl, sl, il = build_sector(k_low, N); Hl = H_pairing_sparse(k_low, g, N, sl, il).toarray()
    wl, vl = np.linalg.eigh(Hl); El = wl[0]; psi_low = vl[:, 0]
    Pm = prolong_matrix(k_low, k_high, N); Phi0 = Pm @ psi_low
    nh, sh, ih = build_sector(k_high, N); Hh = H_pairing_sparse(k_high, g, N, sh, ih).toarray()
    mu = El + mu_buf
    Hle = Pm @ (Hl - mu*np.eye(len(Hl))) @ Pm.T; Hhs = Hh - mu*np.eye(len(Hh))
    psi = Phi0.astype(complex).copy(); dt = T/M
    for m in range(M):
        s = (m + 0.5)/M
        psi = expm(-1j*(np.cos(np.pi*s/2)**2*Hle + np.sin(np.pi*s/2)**2*Hhs)*dt) @ psi
    return psi/np.linalg.norm(psi), Hh

def refine_NK(N, k_high, Ts, k_low=2, M=160, mu_buf=0.6, g=1.0):
    """Overlap and physical energy of the refined state vs adiabatic time T."""
    nl, sl, il = build_sector(k_low, N); Hl = H_pairing_sparse(k_low, g, N, sl, il).toarray()
    wl, vl = np.linalg.eigh(Hl); El = wl[0]; psi_low = vl[:, 0]
    Pm = prolong_matrix(k_low, k_high, N); Phi0 = Pm @ psi_low
    nh, sh, ih = build_sector(k_high, N); Hh = H_pairing_sparse(k_high, g, N, sh, ih).toarray()
    wh, vh = np.linalg.eigh(Hh); Eh = wh[0]; Psih = vh[:, 0]
    mu = El + mu_buf
    Hle = Pm @ (Hl - mu*np.eye(len(Hl))) @ Pm.T; Hhs = Hh - mu*np.eye(len(Hh))
    ov, en = [], []
    for T in Ts:
        psi = Phi0.astype(complex).copy(); dt = T/M
        for m in range(M):
            s = (m + 0.5)/M
            psi = expm(-1j*(np.cos(np.pi*s/2)**2*Hle + np.sin(np.pi*s/2)**2*Hhs)*dt) @ psi
        ov.append(abs(np.vdot(Psih, psi))**2)
        en.append(float(np.real(psi.conj() @ (Hh @ psi))))
    return dict(ov=np.array(ov), en=np.array(en), Eh=Eh, El=El, gap=wh[1]-wh[0],
                ov0=abs(np.vdot(Psih, Phi0))**2, E0=float(np.real(Phi0 @ Hh @ Phi0)))
