import numpy as np
import pairinglib as pl

def test_prolongation_preserves_energy():
    # <Phi0|H_high|Phi0> must equal E_low for the prolonged ground state
    r = pl.refine_NK(4, 4, np.array([0.5]))
    assert abs(r['E0'] - r['El']) < 1e-9

def test_refinement_improves_overlap_and_energy():
    r = pl.refine_NK(4, 4, np.array([0.5, 25.0]))
    assert r['ov0'] < r['ov'][-1]          # overlap improves
    assert r['ov'][-1] > 0.99              # reaches high fidelity
    assert r['en'][-1] >= r['Eh'] - 1e-6   # energy approaches FCI from above
