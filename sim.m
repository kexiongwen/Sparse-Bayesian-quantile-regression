tic
n=500;
p=1000;
BetaTrue = zeros(p,1);
BetaTrue(1)=3;
BetaTrue(2)=1.5;
BetaTrue(5)=2;
BetaTrue(10)=1;
BetaTrue(13)=1;
BetaTrue(19)=0.5;
BetaTrue(26)=-0.5;
BetaTrue(31)=2.0;
BetaTrue(46)=-1.2;
BetaTrue(51)=-1;
Corr=0.5.^toeplitz((0:p-1));
X=mvnrnd(zeros(1,p),Corr,n);
Z=binornd(1,0.8,n,1);
SigmaTrue1=1;
SigmaTrue2=6;
Y=X*BetaTrue+(SigmaTrue1*Z+SigmaTrue2*(1-Z)).*randn([n 1]);
toc

Q=0.5;

%tic
%[beta_sample]=BQR(Y,X,Q);
%toc


tic
[beta_sample]=BQR_GPU(Y,X,Q);
toc


beta_mean=mean(beta_sample,2);






