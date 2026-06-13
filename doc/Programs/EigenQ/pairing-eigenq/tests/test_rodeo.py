import numpy as np
from scipy.linalg import expm
import pairinglib as pl

def test_ancilla_equals_compact_filter():
    rng = np.random.default_rng(0); H = np.diag([0.0, 1.0, 2.5]); n = 3
    v = rng.normal(size=n) + 1j*rng.normal(size=n); v /= np.linalg.norm(v)
    U = expm(-1j*H*0.7)
    a, pa = pl.rodeo_cycle_ancilla(v.copy(), U, 0.7, 0.5)
    b, pb = pl.rodeo_post0(v.copy(), U, 0.7, 0.5)
    assert np.max(np.abs(a - b)) < 1e-12 and abs(pa - pb) < 1e-12

def test_eq2_gaussian_average():
    sigma, dE = 3.0, 0.5
    ts = np.random.default_rng(1).normal(0, sigma, 400000)
    emp = np.mean(np.cos(dE*ts/2)**2)
    ana = 0.5*(1 + np.exp(-dE**2*sigma**2/2))
    assert abs(emp - ana) < 5e-3

def test_rodeo_polishes_refined_state():
    psi, Hh = pl.refine_state(4, 4, 25.0)
    w, V = np.linalg.eigh(Hh); E0, GS = w[0], V[:, 0]
    p0 = abs(np.vdot(GS, psi))**2
    ts = np.random.default_rng(2).normal(0, 3.0, 6)
    F, A = pl.rodeo_track(psi, Hh, ts, E0, GS)
    assert F[-1] > p0 and F[-1] > 0.999      # fidelity improves toward 1
    assert abs(A[-1] - p0) < 5e-3            # acceptance -> p
