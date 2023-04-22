import numpy as np
from scipy.sparse import spdiags
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import invgamma
from scipy.stats import invgauss
from scipy.sparse.linalg import cg


def BQR(Y,X,Q=0.5,M=10000,burn_in=10000):

    N,P=np.shape(X)
    

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

        #Sample b
        b=GXTD@DY+GXTD@np.random.randn(N,1)+np.random.randn(P,1)

        #Solve Preconditioning the linear system by conjugated gradient method
        beta_tilde,_=cg(GXTDXG+sparse.diags(np.ones(P)),b.ravel(),x0=np.zeros(P),tol=1e-2)

        #revert to the solution of the original system
        beta_sample[:,i]=G*beta_tilde

        #Sample lambda
        lam_sample=np.random.gamma(2*P+0.5,((np.abs(beta_sample[:,i])**0.5).sum()+1/a_sample)**-1)
            
        #sample_a
        a_sample=invgamma.rvs(1)*(1+lam_sample)
        
        ink=lam_sample*np.sqrt(np.abs(beta_sample[:,i]))

        #Sample V
        v_sample=2/invgauss.rvs(np.reciprocal(ink))
        
        #Sample tau2
        tau_sample=v_sample/np.sqrt(invgauss.rvs(v_sample/(np.square(ink))))
       
        
        #Sample omega
        omega_sample=invgauss.rvs(2/np.abs(Y-X@beta_sample[:,i:i+1]))/(2*Q*(1-Q))

    MCMC_chain=(beta_sample[:,burn_in:])

    return MCMC_chain