import numpy as np, time
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize

# ===================== Fock-basis (bitstring) machinery =====================
def _bit(s,j,n):  return (s>>(n-1-j))&1
def _flip(s,j,n): return s ^ (1<<(n-1-j))
def _jw_sign(s,j,n):
    c=0
    for k in range(j):
        if (s>>(n-1-k))&1: c+=1
    return 1-2*(c%2)

def build_sector(k,N):
    nq=2*k
    states=sorted(sum(1<<(nq-1-j) for j in c) for c in combinations(range(nq),N))
    index={s:i for i,s in enumerate(states)}
    return nq,states,index

def H_pairing_sparse(k,g,N,states,index,delta=1.0):
    nq=2*k; M=len(states); gh=-0.5*g
    rows,cols,vals=[],[],[]
    for I in states:
        a=index[I]
        rows.append(a);cols.append(a)
        vals.append(sum(delta*(j//2) for j in range(nq) if _bit(I,j,nq)))
        for p in range(k):
            for q in range(k):
                r0,r1=2*q,2*q+1; c0,c1=2*p,2*p+1
                if not _bit(I,r0,nq): continue
                s1=_jw_sign(I,r0,nq); t=_flip(I,r0,nq)
                if not _bit(t,r1,nq): continue
                s2=_jw_sign(t,r1,nq); t=_flip(t,r1,nq)
                if _bit(t,c1,nq): continue
                s3=_jw_sign(t,c1,nq); t=_flip(t,c1,nq)
                if _bit(t,c0,nq): continue
                s4=_jw_sign(t,c0,nq); J=_flip(t,c0,nq)
                rows.append(index[J]);cols.append(a);vals.append(gh*s1*s2*s3*s4)
    H=csr_matrix((vals,(rows,cols)),shape=(M,M))
    return (H+H.T)*0.5

def fci_ground(H):
    M=H.shape[0]
    if M==1: return float(H[0,0])
    if M<=1500: return float(np.linalg.eigvalsh(H.toarray())[0])
    return float(eigsh(H,k=1,which='SA',maxiter=5000)[0][0])

def E_HF(N,g,delta=1.0):
    P=N//2
    return 2.0*delta*sum(range(P)) - 0.5*g*P

# ===================== UCCSD pool + excitation tables =====================
def uccsd_pool(k,N):
    nq=2*k; occ=list(range(N)); virt=list(range(N,nq))
    singles=[(i,a) for i in occ for a in virt if (i%2)==(a%2)]
    doubles=[(i,j,a,b) for (i,j) in combinations(occ,2) for (a,b) in combinations(virt,2)
             if (i%2)+(j%2)==(a%2)+(b%2)]
    return singles,doubles

def exc_single(i,a,nq,states,index):
    Is,Js,sg=[],[],[]
    for I in states:
        if not _bit(I,i,nq) or _bit(I,a,nq): continue
        s1=_jw_sign(I,i,nq); t=_flip(I,i,nq); s2=_jw_sign(t,a,nq); J=_flip(t,a,nq)
        Is.append(index[I]);Js.append(index[J]);sg.append(s1*s2)
    return (np.array(Is),np.array(Js),np.array(sg,float))

def exc_double(i,j,a,b,nq,states,index):
    Is,Js,sg=[],[],[]
    for I in states:
        if not _bit(I,i,nq): continue
        s1=_jw_sign(I,i,nq); t=_flip(I,i,nq)
        if not _bit(t,j,nq): continue
        s2=_jw_sign(t,j,nq); t=_flip(t,j,nq)
        if _bit(t,b,nq): continue
        s3=_jw_sign(t,b,nq); t=_flip(t,b,nq)
        if _bit(t,a,nq): continue
        s4=_jw_sign(t,a,nq); J=_flip(t,a,nq)
        Is.append(index[I]);Js.append(index[J]);sg.append(s1*s2*s3*s4)
    return (np.array(Is),np.array(Js),np.array(sg,float))

def apply_exc(psi,theta,arr):
    Is,Js,sg=arr
    if Is.size==0: return
    c,s=np.cos(theta),np.sin(theta)
    a=psi[Is].copy(); b=psi[Js].copy()
    psi[Is]=c*a-sg*s*b; psi[Js]=sg*s*a+c*b

def tau_apply(psi,arr):
    Is,Js,sg=arr; out=np.zeros_like(psi)
    if Is.size: out[Is]=-sg*psi[Js]; out[Js]=sg*psi[Is]
    return out

# ===================== generalized CCD (week47) =====================
def init_pairing_v(g,pnum,hnum):
    v_pppp=np.zeros((pnum,)*4); v_pphh=np.zeros((pnum,pnum,hnum,hnum)); v_hhhh=np.zeros((hnum,)*4)
    gv=-0.5*g
    for a in range(0,pnum,2):
        for b in range(0,pnum,2):
            v_pppp[a,a+1,b,b+1]=gv;v_pppp[a+1,a,b,b+1]=-gv;v_pppp[a,a+1,b+1,b]=-gv;v_pppp[a+1,a,b+1,b]=gv
    for a in range(0,pnum,2):
        for i in range(0,hnum,2):
            v_pphh[a,a+1,i,i+1]=gv;v_pphh[a+1,a,i,i+1]=-gv;v_pphh[a,a+1,i+1,i]=-gv;v_pphh[a+1,a,i+1,i]=gv
    for j in range(0,hnum,2):
        for i in range(0,hnum,2):
            v_hhhh[j,j+1,i,i+1]=gv;v_hhhh[j+1,j,i,i+1]=-gv;v_hhhh[j,j+1,i+1,i]=-gv;v_hhhh[j+1,j,i+1,i]=gv
    return v_pppp,v_pphh,v_hhhh
def init_pairing_fock(delta,g,pnum,hnum):
    dv,gv=0.5*delta,-0.5*g; f_pp=np.zeros((pnum,pnum));f_hh=np.zeros((hnum,hnum))
    for i in range(0,hnum,2): f_hh[i,i]=dv*i+gv; f_hh[i+1,i+1]=dv*i+gv
    for a in range(0,pnum,2): f_pp[a,a]=dv*(hnum+a); f_pp[a+1,a+1]=dv*(hnum+a)
    return f_pp,f_hh
def init_t2(v_pphh,f_pp,f_hh):
    pn,hn=len(f_pp),len(f_hh); t2=np.zeros((pn,pn,hn,hn))
    for i in range(hn):
        for j in range(hn):
            for a in range(pn):
                for b in range(pn):
                    t2[a,b,i,j]=v_pphh[a,b,i,j]/(f_hh[i,i]+f_hh[j,j]-f_pp[a,a]-f_pp[b,b])
    return t2
def ccd_iter(v_pppp,v_pphh,v_hhhh,f_pp,f_hh,t2):
    pn,hn=len(f_pp),len(f_hh)
    Hb=(v_pphh+np.einsum('bc,acij->abij',f_pp,t2)-np.einsum('ac,bcij->abij',f_pp,t2)
        -np.einsum('abik,kj->abij',t2,f_hh)+np.einsum('abjk,ki->abij',t2,f_hh)
        +0.5*np.einsum('abcd,cdij->abij',v_pppp,t2)+0.5*np.einsum('abkl,klij->abij',t2,v_hhhh))
    chh=0.5*np.einsum('cdkl,cdjl->kj',v_pphh,t2)
    Hb-=(np.einsum('abik,kj->abij',t2,chh)-np.einsum('abik,kj->abji',t2,chh))
    cpp=-0.5*np.einsum('cdkl,bdkl->cb',v_pphh,t2)
    Hb+=(np.einsum('acij,cb->abij',t2,cpp)-np.einsum('acij,cb->baij',t2,cpp))
    chhhh=0.5*np.einsum('cdkl,cdij->klij',v_pphh,t2)
    Hb+=0.5*np.einsum('abkl,klij->abij',t2,chhhh)
    cphph=0.5*np.einsum('cdkl,dblj->bkcj',v_pphh,t2)
    Hb+=(np.einsum('bkcj,acik->abij',cphph,t2)-np.einsum('bkcj,acik->baij',cphph,t2)
         -np.einsum('bkcj,acik->abji',cphph,t2)+np.einsum('bkcj,acik->baji',cphph,t2))
    t2n=np.zeros_like(t2)
    for i in range(hn):
        for j in range(hn):
            for a in range(pn):
                for b in range(pn):
                    t2n[a,b,i,j]=t2[a,b,i,j]+Hb[a,b,i,j]/(f_hh[i,i]+f_hh[j,j]-f_pp[a,a]-f_pp[b,b])
    return t2n
def run_ccd(k,g,N,delta=1.0,niter=300,mix=0.5,tol=1e-13):
    hnum=N; pnum=2*k-N; Eref=E_HF(N,g,delta)
    if pnum==0: return Eref,0.0
    vpppp,vpphh,vhhhh=init_pairing_v(g,pnum,hnum); fpp,fhh=init_pairing_fock(delta,g,pnum,hnum)
    t2=init_t2(vpphh,fpp,fhh); erg=0.25*np.einsum('abij,abij',vpphh,t2)
    for _ in range(niter):
        t2n=ccd_iter(vpppp,vpphh,vhhhh,fpp,fhh,t2); ergn=0.25*np.einsum('abij,abij',vpphh,t2n)
        t2=mix*t2n+(1-mix)*t2
        if abs(ergn-erg)<tol: erg=ergn;break
        erg=ergn
    return Eref+erg,erg

# ===================== Trotterized UCCSD-VQE =====================
def setup_uccsd(k,N,delta=1.0):
    nq,states,index=build_sector(k,N)
    singles,doubles=uccsd_pool(k,N)
    arrs=([exc_single(i,a,nq,states,index) for (i,a) in singles]
          +[exc_double(i,j,a,b,nq,states,index) for (i,j,a,b) in doubles])
    hf=index[sum(1<<(nq-1-j) for j in range(N))]
    return dict(nq=nq,states=states,index=index,singles=singles,doubles=doubles,
                arrs=arrs,nS=len(singles),P=len(arrs),M=len(states),hf=hf)

def uccsd_vqe(setup,H,n_trotter=1,x0=None,tol=(1e-14,1e-12),maxiter=3000):
    arrs=setup['arrs']; P=setup['P']; M=setup['M']; hf=setup['hf']
    seq=list(range(P))*n_trotter; S=len(seq)
    def energy(p):
        ang=np.repeat(p/n_trotter,1); psi=np.zeros(M);psi[hf]=1.0
        for step,kk in enumerate(seq): apply_exc(psi,p[kk]/n_trotter,arrs[kk])
        return float(psi@(H@psi))
    def grad(p):
        psi=np.zeros(M);psi[hf]=1.0; st=[psi.copy()]
        for step,kk in enumerate(seq): apply_exc(psi,p[kk]/n_trotter,arrs[kk]); st.append(psi.copy())
        Hpsi=H@st[-1]; gstep=np.zeros(S)
        for step in range(S):
            kk=seq[step]; v=tau_apply(st[step+1],arrs[kk])
            for s2 in range(step+1,S): apply_exc(v,p[seq[s2]]/n_trotter,arrs[seq[s2]])
            gstep[step]=2.0*float(v@Hpsi)
        g=np.zeros(P)
        for step in range(S): g[seq[step]]+=gstep[step]/n_trotter
        return g
    def energy0():
        psi=np.zeros(M);psi[hf]=1.0; return float(psi@(H@psi))
    if P==0:
        return energy0(), np.zeros(0), (lambda p: energy0()), (lambda p: np.zeros(0))
    if x0 is None: x0=np.zeros(P)
    res=minimize(energy,x0,jac=grad,method='L-BFGS-B',
                 options={'ftol':tol[0],'gtol':tol[1],'maxiter':maxiter})
    return res.fun,res.x,energy,grad
