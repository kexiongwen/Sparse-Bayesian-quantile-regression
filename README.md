# Bayesian quantile regression with $L_{\frac{1}{2}}$ prior

##  Model setting	

Assume that $y_{i}=x_{i}^{T}\beta+\epsilon_{i}$ with $\epsilon_{i}$ being i.i.d random variables from the skewed Laplace distribution with density
$$
f(\epsilon)=q(1-q) \exp[-\rho_{q}(\epsilon)]
$$
for $q \in (0,1)$ . Then the joint distribution of $Y=(y_{1},...,y_{n})$ given $X=(x_{1},...,x_{n})$ is
$$
f(Y\mid X, \beta,\sigma)=q^{n}(1-q)^{n}\exp\left\{-\sum_{i=1}^{n}\rho_{q}(y_{i}-x_{i}^{T}\beta) \right\}
$$
Since the skewed Laplace distribution can be represented as a scale mixture of normals, we have
$$
y_{i}=x_{i}^{T}\beta+(\theta_{1}w_{i}+\theta_{2}z_{i}\sqrt{w_{i}})
$$
where $\theta_{1}=\frac{1-2q}{q(1-q)}$,$\theta_{2}=\sqrt{\frac{2}{q(1-q)}}$,$z_{i}\sim N(0,1)$ and $w_{i} \sim \mathrm{Exp}(1)$.

We consider a $L_{\frac{1}{2}}$ prior on $\beta_{j}$ such that 
$$
\pi(\beta_{j} \mid \sigma, \lambda)\propto \exp[-\lambda |\beta_{j}|^{\frac{1}{2}}]
$$
with hyper-prior 
$$
\frac{1}{\sqrt{\lambda}} \sim \mathrm{Cauchy_{+}}(0,1),
$$

## Partially collapsed Gibbs sampling

1. Sample $\beta \mid \lambda,\tau^{2},w  \sim N(\mu,\Sigma)$

where $\Sigma= (X^{T}DX+\lambda^{4}\Lambda^{-2})^{-1}$ , $\mu=\Sigma X^{T}D(Y-\frac{(1-2q)}{q(1-q)} W)$ ,  $D=\frac{q(1-q)}{2}\mathrm{Diag}(w_{i}^{-1})$ and $\Lambda=\mathrm{Diag}(\tau^{2})$

2. Sample $\lambda \mid \beta, b \sim \mathrm{Gamma}(2P+0.5,\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2}}+1/b)$
3. Sample $\frac{1}{v_{j}} \mid \beta_{j}, \lambda
   \sim \mathrm{InvGaussian}\left (\sqrt{\frac{1}{4\lambda^{2}|\beta_{j}|}},\frac{1}{2}\right)$ for $j=1,...,p$
4. Sample $\frac{1}{{\tau}_{j}^{2}} \mid \lambda,\beta_{j},v_{j} 
   \sim  \mathrm{InvGaussian}\left (\frac{1}{\lambda^{2}{v}_{j}|\beta_{j}|},\frac{1}{{v}_{j}^{2}}\right)$ for $j=1,...,p$
5. Sample $\frac{1}{w_{i}} \mid \beta, \sigma, y_{i} \sim \mathrm{InvGaussian}\left(\frac{1}{q(1-q)|y_{i}-x_{i}^{T}\beta|},\frac{1}{2q(1-q)}\right)$ for $i=1,...,n$
6. Sample $b \mid \lambda \sim \mathrm{InvGamma}(1,1+\lambda)$



## Scalable PCG sampler via conjugated gradient, prior precondition and sparse linear system approximation

The similar PCG sampling scheme for $L_\frac{1}{2}$ prior can also be applied to the likelihood which can be decomposed as scale mixture of Gaussian. However, in the “large *N* and large *P*” setting, the required posterior computation for $\beta$ encounters a bottleneck at repeated sampling from a high-dimensional Gaussian distribution, whose covariance matrix is expensive to compute and factorize. For example, in this quantile regression case, we have $\Sigma= (X^{T}DX+\lambda^{4}\Lambda^{-2})^{-1}$ . In the article, https://www.tandfonline.com/doi/epdf/10.1080/01621459.2022.2057859?needAccess=true&role=button, the authors show that they can generate a random vector $b$ with low computation cost, such that the solution to the linear system $\Sigma^{-1}\beta=b$ has the desire Gaussian distribution. This linear system can be solved by conjugate gradient algorithm, which doesn't involve explicity factorization of $\Sigma^{-1}$.Motived by the JOB-approximation from https://www.jmlr.org/papers/volume21/19-536/19-536.pdf,  we propose a sparse linear system approximation for $\Sigma^{-1}$, which can further speed up their algorithm.



### Conjugated gradient

The following procedure generates a sample $\beta$ from $\pi(\beta \mid \lambda,\tau^{2},\omega)$:

1. Generate $b \sim \mathcal{N}\left(X^{T} D \tilde{y}, \Phi\right)$ by sampling independent Gaussian vectors  $\eta \sim \mathcal{N}\left(0, I_{n}\right)$ and $\delta \sim \mathcal{N}\left(0, I_{p}\right)$ and then setting
   $$
   \begin{equation}
   b=X^{T}D\tilde{Y}+X^{T} D^{1 / 2} \eta+\lambda^{2}\tau^{-1} \odot \delta
   \end{equation}
   $$
   where $\Phi=X^{T} D X+\lambda^{4}\Lambda^{-1}$.

   

2. Solve the following linear system for $\beta$
   $$
   \Phi \boldsymbol{\beta}=b
   $$

Since $\Phi$ is symmetric and positive-definite, solving the linear system above can be further speed up by using conjugated gradient method. Given an initial guess of $\beta$, which may be taken as $0$ or $\beta^{(t-1)}$ for example, conjugated gradient method generates a sequence $\left\{\beta_{k}\right\}$,  $k=1,2,...$ of increasingly accurate approximations to the solution.

### Prior preconditioning

To accelerate the  convergence of conjugated gradient,  the global and local shrinkage parameters will be used to precondition the linear system $\Phi \boldsymbol{\beta}=b$  In high-dimensional and very sparse setting,  the covariance matrix $(X^{T} D X+\lambda^{4}\Lambda^{-1})^{-1}$ for the conditional posterior of $\beta$  will near to singular. The prior preconditioning approach can also improve the numerical stable of the PCG sampler.

A preconditioner is a positive definite matrix $M$ chosen such that the preconditioned system
$$
\tilde{\Phi} \tilde{\beta}=\tilde{b} \quad \text{for} \quad \tilde{\Phi}=M^{-1 / 2}\Phi M^{-1 / 2} \quad \text{and} \quad \tilde{b}=M^{-1 / 2} b
$$
where $M=\lambda^{4}\Lambda^{-1}$. By setting $\beta=M^{-1/2}\tilde{\beta}$,  we obatin the solution of the original linear system. 



The prior-preconditioned matrix  is given by


$$
\tilde{\Phi}=\lambda^{-4} \Lambda^{1/2} X^{T} D X \Lambda^{1/2}+I_{p}
$$
The prior-preconditioned vector is given by


$$
\begin{equation}\label{eq:b_tilde}
\tilde{b}=\lambda^{-2}\Lambda^{1/2}X^{T}D\tilde{Y}+\lambda^{-2}\Lambda^{1/2} X^{T} D^{1 / 2} \eta+ \delta
\end{equation}
$$
The $(i, j)$th entry of the matrix $\tilde{\Phi}$ is given by


$$
\tilde{\Phi}_{ij}= \begin{cases} \left(\lambda^{-2} \tau_i\right)\left(\lambda^{-2} \tau_{j}\right)\left(X^{T}D X\right)_{i j} & \text { if } i \neq j \\ \left(\lambda^{-4} \tau_{i}^{2}\right)\left(X^{T}D X\right)_{i i} +1 & \text { if } i=j \end{cases}
$$


### Sparse linear system approximation

If $\beta_{i}$ or $\beta_{j}$ is identified as noise, then $\lambda^{-2} \tau_{i} \approx 0$ or $\lambda^{-2} \tau_{j} \approx 0$.  We have $\tilde{\Phi}_{ij} \approx 0$ or $\tilde{\Phi}_{ii} \approx 1$. By using a user-deﬁned thresholding parameter $\Delta$, we can have sparse approximation for $\tilde{\Phi}$, such that


$$
\begin{aligned}
{\tilde{\Phi}_{\Delta}}_{ij}= &
\begin{cases} 
\left(\lambda^{-2} \tau_i\right)\left(\lambda^{-2} \tau_{j}\right)\left(X^{T}D X\right)_{i j} & \text { if }  \lambda^{-2} \tau_i>\Delta \,\, \text{or}\,\, \lambda^{-2} \tau_{j}>\Delta\\ 
0 & \text { else } 
\end{cases}\\
{\tilde{\Phi}_{\Delta}}_{ii}= &
\begin{cases} 
\left(\lambda^{-4} \tau_{i}^{2}\right)\left(X^{T}D X\right)_{i i}+1 & \quad\,\,\, \text { if }  \lambda^{-2} \tau_i>\Delta\\
1 & \quad\,\, \,\text { else } 
\end{cases}
\end{aligned}
$$


Therefore, we obtain a three-step procedure to sample the condition posterior of $\beta$:



1. Generate $\tilde{b} \sim \mathcal{N}\left(\lambda^{-2}\Lambda^{1/2}X^{T} D \tilde{Y}, \tilde{\Phi}\right)$ by using equation $(\ref{eq:b_tilde})$.

   

2. Use conjugated gradient method to solve the following linear system for $\tilde{\beta}_{\Delta}$:

   
   $$
   \tilde{\Phi}_{\Delta}\tilde{\beta}_{\Delta}=\tilde{b}
   $$
   

3. Setting $\beta_{\Delta}=\lambda^{-2}\Lambda^{1/2}\tilde{\beta}_{\Delta}$, then $\beta_{\Delta} \sim \mathcal{N}\left(\lambda^{-2}\Lambda^{1/2} \tilde{\Phi}_{\Delta}^{-1} X^{T} D \tilde{y}, \lambda^{-4}\Lambda\tilde{\Phi}_{\Delta}^{-1}\tilde{\Phi}\tilde{\Phi}_{\Delta}^{-1}\right)$.