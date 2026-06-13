import numpy as np
import pairinglib as pl

def test_benchmarks_k2_k3():
    for k, ref in [(2, 1.000000), (3, 0.794697)]:
        su = pl.setup_uccsd(k, 4)
        H = pl.H_pairing_sparse(k, 1.0, 4, su['states'], su['index'])
        assert abs(pl.fci_ground(H) - ref) < 1e-5
        assert abs(pl.run_ccd(k, 1.0, 4)[0] - ref) < 1e-5
        assert abs(pl.uccsd_vqe(su, H)[0] - ref) < 1e-5

def test_uccsd_variational_k4():
    su = pl.setup_uccsd(4, 4)
    H = pl.H_pairing_sparse(4, 1.0, 4, su['states'], su['index'])
    fci = pl.fci_ground(H)
    assert abs(fci - 0.635548) < 1e-5
    assert pl.uccsd_vqe(su, H)[0] >= fci - 1e-9   # variational

def test_hf_energy():
    assert abs(pl.E_HF(4, 1.0) - 1.0) < 1e-12     # 2 - g
