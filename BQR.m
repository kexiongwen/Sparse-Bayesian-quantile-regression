function[beta_sample]=BQR(Y,X,Q)

M=10000;
burn_in=10000;
S=size(X);
T1=1e-2;
T2=1e-4;
c1=(0.5*Q*(1-Q)).^0.5;
c2=(1-2*Q)/(Q*(1-Q));
beta_sample=zeros(S(2),M+burn_in);
tau_sample=ones(S(2),1);
v_sample=ones(S(2),1);
omega_sample=ones(S(1),1);
a_sample=1;
lam_sample=1;

for i=2:(M+burn_in)
    
    %Sample beta
    %Prior preconditioning matrix from global-local shrinkage
    G=(tau_sample)./lam_sample.^2;
    
    %Weight
    D=c1*sqrt(omega_sample);

    %Preconditioning feature matrix
    XTD=X'.*D';
    GXTD=G.*XTD;
    DY=D.*(Y-c2./omega_sample);

    %Preconditioning covariance matrix
    GXTDXG=GXTD*GXTD';

    Mask1=G<T1;

    %Sample b
    b=GXTD*DY+GXTD*randn(S(1),1)+randn(S(2),1);

    %Solve Preconditioning the linear system by conjugated gradient method
    beta_tilde=cgs(sparse(GXTDXG.*(1-Mask1*Mask1')+speye(S(2))),b,1e-3);

    %revert to the solution of the original system
    beta_sample(:,i)=G.*beta_tilde;

    % Sampling lambda
    lam_sample=gamrnd(2*S(2)+0.5,1./(sum(sqrt(abs(beta_sample(:,i))))+1./a_sample));

    % Sampling a
    a_sample=1./gamrnd(1,1./(1+lam_sample));

    ink=lam_sample.^2.*abs(beta_sample(:,i));
    Mask2=ink<T2;

    % Sampling V
    v_sample(~Mask2)=2./random('InverseGaussian',1./sqrt(ink(~Mask2)),1);
    v_sample(Mask2)=gamrnd(0.5,4*ones(sum(Mask2),1));

    % Sampling tau
    tau_sample(~Mask2)=v_sample(~Mask2)./sqrt(random('InverseGaussian',v_sample(~Mask2)./ink(~Mask2),1));
    tau_sample(Mask2)=sqrt(gamrnd(0.5,2*v_sample(Mask2).^2));

    % Sampling omega
    omega_sample=random('InverseGaussian',2./abs(Y-X*beta_sample(:,i)),1)/(2*Q*(1-Q));

end

beta_sample=beta_sample(:,burn_in+1:end);

end