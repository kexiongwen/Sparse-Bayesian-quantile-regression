import numpy as np
from scipy.sparse import spdiags
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import invgamma
from scipy.stats import invgauss
from scipy.sparse.linalg import cg


def BQR(Y,X,Q=0.5,M=10000,burn_in=10000):

    N,P=np.shape(X)
    T1=1e-2
    T2=1e-3

    #Initialization
    beta_sample=np.zeros((P,M+burn_in))
    beta_tilde=np.ones(P)
    tau_sample=np.ones(P)  
    v_sample=np.ones(P)
    a_sample=1
    lam_sample=1
    omega_sample=np.ones((N,1))
    c1=(0.5*Q*(1-Q))**0.5
    c2=(1-2*Q)/(Q*(1-Q))
    b=np.zeros((P,1))
    Mask1=np.zeros((P,1))
    Mask2=np.zeros(P)

    for i in range(1,M+burn_in):

        #Sample beta

        #Prior preconditioning matrix from global-local shrinkage
        G=(tau_sample)/lam_sample**2
         
        #Weight
        D=c1*np.sqrt(omega_sample)

        #Preconditioning feature matrix
        XTD=X.T*D.T         
        GXTD=G.reshape(-1,1)*XTD           
        DY=D*(Y-c2/omega_sample)
        
        #Preconditioning covariance matrix
        GXTDXG=GXTD@GXTD.T

        Mask1[:,0]=(G<T1).astype(float)

        #Sample b
        b=GXTD@DY+GXTD@np.random.randn(N,1)+np.random.randn(P,1)

        #Solve Preconditioning the linear system by conjugated gradient method
        beta_tilde,_=cg(csr_matrix(GXTDXG*(1-Mask1@Mask1.T)+sparse.diags(np.ones(P))),b.ravel(),x0=np.zeros(P),tol=1e-4)

        #revert to the solution of the original system
        beta_sample[:,i]=G*beta_tilde

        #Sample lambda
        lam_sample=np.random.gamma(2*P+0.5,((np.abs(beta_sample[:,i])**0.5).sum()+1/a_sample)**-1)
            
        #sample_a
        a_sample=invgamma.rvs(1)*(1+lam_sample)
        
        ink=lam_sample*np.sqrt(np.abs(beta_sample[:,i]))

        Mask2=ink<T2
    
        #Sample V
        v_sample[~Mask2]=2/invgauss.rvs(np.reciprocal(ink[~Mask2]))
        v_sample[Mask2]=np.random.gamma(0.5,4*np.ones_like(v_sample[Mask2]))

        #Sample tau2
        tau_sample[~Mask2]=v_sample[~Mask2]/np.sqrt(invgauss.rvs(v_sample[~Mask2]/(np.square(ink[~Mask2]))))
        tau_sample[Mask2]=np.sqrt(np.random.gamma(0.5,2*np.square(v_sample[Mask2])))
        
        #Sample omega
        omega_sample=invgauss.rvs(2/np.abs(Y-X@beta_sample[:,i:i+1]))/(2*Q*(1-Q))

    MCMC_chain=(beta_sample[:,burn_in:])

    return MCMC_chain