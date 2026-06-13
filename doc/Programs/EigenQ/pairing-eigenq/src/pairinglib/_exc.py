import numpy as np, itertools
from scipy.linalg import expm
from ._gates import pauli_exp
I2=np.eye(2,dtype=complex); Xm=np.array([[0,1],[1,0]],complex)
Ym=np.array([[0,-1j],[1j,0]],complex); Zm=np.array([[1,0],[0,-1]],complex)
_P={'I':I2,'X':Xm,'Y':Ym,'Z':Zm}
def kron_n(ops):
    r=ops[0]
    for o in ops[1:]: r=np.kron(r,o)
    return r
def jw_create(j,n): return kron_n([Zm if l<j else ((Xm-1j*Ym)/2 if l==j else I2) for l in range(n)])
def jw_ann(j,n): return jw_create(j,n).conj().T
def pauli_mat(P,n): return kron_n([_P[P.get(q,'I')] for q in range(n)])
def between(p,q):
    lo,hi=min(p,q),max(p,q); return list(range(lo+1,hi))

def _pauli_terms_of_generator(T,active,zset,n,npauli):
    """Extract (Pdict, coeff_imag) for odd-Y(=npauli odd) patterns; coeff so that
       T = sum (1j*coeff) * Pauli(Pdict).  Returns list and reconstruction error."""
    base={q:'Z' for q in zset}
    patterns=[c for c in itertools.product('XY',repeat=len(active))
              if c.count('Y')%2==1]
    terms=[]; recon=np.zeros_like(T)
    for pat in patterns:
        P=dict(base)
        for q,pa in zip(active,pat): P[q]=pa
        M=pauli_mat(P,n)
        coeff=np.trace(T@M)/(2**n)          # T = sum coeff_k P_k ; coeff complex
        if abs(coeff)>1e-9:
            terms.append((P,np.imag(coeff)))   # generator is i*(real) -> coeff purely imaginary
            recon=recon+coeff*M
    err=np.max(np.abs(recon-T))
    return terms,err

def make_single(i,a,n):
    G=jw_create(a,n)@jw_ann(i,n); T=G-G.conj().T
    terms,err=_pauli_terms_of_generator(T,[i,a],between(i,a),n,1)
    assert err<1e-9,f"single recon err {err}"
    return terms      # list of (Pdict, b_k) with T = sum i*b_k P_k

def make_double(i,j,a,b,n):
    G=jw_create(a,n)@jw_create(b,n)@jw_ann(j,n)@jw_ann(i,n); T=G-G.conj().T
    zset=set(between(i,j))|set(between(a,b))
    terms,err=_pauli_terms_of_generator(T,[i,j,a,b],zset,n,1)
    assert err<1e-9,f"double recon err {err}"
    return terms

def apply_exc_circuit(psi,theta,terms,n):
    """exp(theta*T) with T=sum i*b_k P_k (commuting) = prod exp(i*theta*b_k P_k)
       and exp(i*phi P)=pauli_exp(psi,-2*phi,P,n)?  pauli_exp does e^{-i phi/2 P}.
       exp(i*theta*b_k*P)=e^{-i*(-2 theta b_k)/2 *P} -> phi=-2*theta*b_k."""
    for P,bk in terms:
        psi=pauli_exp(psi,-2.0*theta*bk,P,n)
    return psi

if __name__=='__main__':
    rng=np.random.default_rng(5); n=6
    # singles
    for _ in range(5):
        i,a=sorted(rng.choice(n,2,replace=False))
        terms=make_single(i,a,n); th=rng.uniform(-1.5,1.5)
        G=jw_create(a,n)@jw_ann(i,n); T=G-G.conj().T
        psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n); psi/=np.linalg.norm(psi)
        d=np.max(np.abs(apply_exc_circuit(psi.copy(),th,terms,n)-expm(th*T)@psi))
        print(f"single ({i}->{a}) #terms={len(terms)} diff={d:.1e}")
    # doubles, various orderings
    for (i,j,a,b) in [(0,1,4,5),(0,1,2,3),(0,2,4,5),(1,3,4,5),(0,3,2,5)]:
        terms=make_double(i,j,a,b,n); th=rng.uniform(-1.5,1.5)
        G=jw_create(a,n)@jw_create(b,n)@jw_ann(j,n)@jw_ann(i,n); T=G-G.conj().T
        psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n); psi/=np.linalg.norm(psi)
        d=np.max(np.abs(apply_exc_circuit(psi.copy(),th,terms,n)-expm(th*T)@psi))
        ncnot=sum(2*(len(P)-1) for P,_ in terms)
        print(f"double ({i},{j}->{a},{b}) #terms={len(terms)} CNOTs={ncnot} diff={d:.1e}")
