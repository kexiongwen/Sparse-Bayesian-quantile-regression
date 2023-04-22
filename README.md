# Bayesian quantile regression with $L_{\frac{1}{2}}$ prior

##  Model setting	

Assume that $y_{i}=x_{i}^{T}\beta+\epsilon_{i}$ with $\epsilon_{i}$ being i.i.d random variables from the skewed Laplace distribution with density


$$
f(\epsilon)=q(1-q) \exp[-\rho_{q}(\epsilon)]
$$

for $q \in (0,1)$ . Then the joint distribution of $Y=(y_{1},...,y_{n})$ given $X=(x_{1},...,x_{n})$ is



$$
f(Y\mid X, \beta,\sigma)=q^{n}(1-q)^{n}\exp\left[-\sum_{i=1}^{n}\rho_{q}(y_{i}-x_{i}^{T}\beta) \right]
$$



Since the skewed Laplace distribution can be represented as a scale mixture of normals, we have



$$
y_{i}=x_{i}^{T}\beta+(\theta_{1}w_{i}+\theta_{2}z_{i}\sqrt{w_{i}})
$$



where $\theta_{1}=\frac{1-2q}{q(1-q)}$,  $\theta_{2}=\sqrt{\frac{2}{q(1-q)}}$,  $z_{i}\sim N(0,1)$ and $w_{i} \sim \mathrm{Exp}(1)$.



We consider a $L_{\frac{1}{2}}$ prior on $\beta_{j}$ such that 


$$
\pi(\beta_{j} \mid \sigma, \lambda)\propto \exp[-\lambda |\beta_{j}|^{\frac{1}{2}}]
$$


with hyper-prior 


$$
\frac{1}{\sqrt{\lambda}} \sim \mathrm{Cauchy_{+}}(0,1),
$$



## Partially Collapsed Gibbs Sampling

1. Sample $\beta \mid \lambda,\tau^{2},w  \sim N(\mu,\Sigma)$



where $\Sigma= (X^{T}DX+\lambda^{4}\Lambda^{-2})^{-1}$ , $\mu=\Sigma X^{T}D(Y-\frac{(1-2q)}{q(1-q)} W)$ ,  $D=\frac{q(1-q)}{2}\mathrm{Diag}(w_{i}^{-1})$ and $\Lambda=\mathrm{Diag}(\tau^{2})$



2. Sample $\lambda \mid \beta, b \sim \mathrm{Gamma}(2p+0.5,\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2}}+1/b)$

   

3. Sample $\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right), \quad j=1, \ldots, p$
   
   
   
4. Sample $\frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right), \quad j=1, \ldots, p$
   
   
   
5. Sample $\frac{1}{w_{i}} \mid \beta, \sigma, y_{i} \sim \mathrm{InvGaussian}\left(\frac{1}{q(1-q)|y_{i}-x_{i}^{T}\beta|},\frac{1}{2q(1-q)}\right), \quad i=1,\dots,N$

   

6. Sample $b \mid \lambda \sim \mathrm{InvGamma}(1,1+\lambda)$



## Scalable PCG sampler via conjugated gradient, prior precondition and sparse linear system approximation

The similar PCG sampling scheme for $L_\frac{1}{2}$ prior can also be applied to the likelihood which can be decomposed as scale mixture of Gaussian. However, in the “large *N* and large *P*” setting, the required posterior computation for $\beta$ encounters a bottleneck at repeated sampling from a high-dimensional Gaussian distribution, whose covariance matrix is expensive to compute and factorize. For example, in this quantile regression case, we have  $\Sigma= (X^{T}DX+\lambda^{4}\Lambda^{-2})^{-1}$ . In the article, https://www.tandfonline.com/doi/epdf/10.1080/01621459.2022.2057859?needAccess=true&role=button, the authors show that they can generate a random vector $b$ with low computation cost, such that the solution to the linear system  $\Sigma^{-1}\beta=b$  has the desire Gaussian distribution. This linear system can be solved by conjugate gradient algorithm, which doesn't involve explicity factorization of $\Sigma^{-1}$



### Conjugated gradient

The following procedure generates a sample $\beta$ from $\pi(\beta \mid \lambda,\tau^{2},\omega)$:

1. Generate $b \sim \mathcal{N}\left(X^{T} D \tilde{y}, \Sigma^{-1}\right)$ by sampling independent Gaussian vectors  $\eta \sim \mathcal{N}\left(0, I_{n}\right)$ and $\delta \sim \mathcal{N}\left(0, I_{p}\right)$ and then setting

   


$$
b=X^{T}D\tilde{Y}+X^{T} D^{1 / 2} \eta+\lambda^{2}\tau^{-1} \odot \delta
$$

   

2. Solve the following linear system for $\beta$ :


$$
{\Sigma}^{-1}{\beta}=b
$$

Since  $\Sigma^{-1}$  is symmetric and positive-definite, solving the linear system above can be further speed up by using conjugated gradient method. Given an initial guess of $\beta$, which may be taken as $0$ or $\beta^{(t-1)}$.  



### Prior preconditioning

To accelerate the  convergence of conjugated gradient,  the global and local shrinkage parameters will be used to precondition the linear system $\Sigma^{-1} \beta=b$  In high-dimensional and very sparse setting,  the covariance matrix $\Sigma$ for the conditional posterior of $\beta$  will near to singular. The prior preconditioning approach can also improve the numerical stable of the PCG sampler.

A preconditioner is a positive definite matrix $M$ chosen such that the preconditioned system


$$
\tilde{\Sigma}^{-1} \tilde{\beta}=\tilde{b} \quad \text{for} \quad \tilde{\Sigma}^{-1}=M^{-1 / 2}\Sigma^{-1} M^{-1 / 2} \quad \text{and} \quad \tilde{b}=M^{-1 / 2} b
$$


where $M=\lambda^{4}\Lambda^{-1}$. By setting $\beta=M^{-1/2}\tilde{\beta}$,  we obatin the solution of the original linear system. 



The prior-preconditioned matrix  is given by



$$
\tilde{\Sigma}^{-1}=\lambda^{-4} \Lambda^{1/2} X^{T} D X \Lambda^{1/2}+I_{p}
$$



The prior-preconditioned vector is given by



$$
\tilde{b}=\lambda^{-2}\Lambda^{1/2}X^{T}D\tilde{Y}+\lambda^{-2}\Lambda^{1/2} X^{T} D^{1 / 2} \eta+ \delta
$$



The $(i, j)$  entry of the matrix  $\tilde{\Sigma}^{-1}$  is given by



$$
\tilde{\Sigma}_{i j}^{-1}= 
\begin{cases}
\left(\lambda^{-2} \tau_i\right)\left(\lambda^{-2} \tau_j\right)\left(X^T D X\right)_{i j} & \text { if } i \neq j \\ 
\left(\lambda^{-4} \tau_i^2\right)\left(X^T D X\right)_{i i}+1 & \text { if } i=j
\end{cases}
$$



### 

## Reference

```
@article{nishimura2022prior,
  title={Prior-Preconditioned Conjugate Gradient Method for Accelerated Gibbs Sampling in “Large n, Large p” Bayesian Sparse Regression},
  author={Nishimura, Akihiko and Suchard, Marc A},
  journal={Journal of the American Statistical Association},
  pages={1--14},
  year={2022},
  publisher={Taylor \& Francis}
}
```

```
@article{ke2021bayesian,
  title={Bayesian $ L_\frac{1}{2}$ regression},
  author={Ke, Xiongwen and Fan, Yanan},
  journal={arXiv preprint arXiv:2108.03464},
  year={2021}
}
```

