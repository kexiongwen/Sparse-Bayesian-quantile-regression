tic
n=100;
p=1000;
BetaTrue = zeros(p,1);
BetaTrue(1)=2;
BetaTrue(2)=2;
BetaTrue(3)=2;
BetaTrue(4)=2;
BetaTrue(5)=2;
SigmaTrue=1;
Corr=0.5.^toeplitz((0:p-1));
X=mvnrnd(zeros(1,p),Corr,n);
Y=X*BetaTrue+SigmaTrue.*randn([n 1]);
toc

Q=0.5;

tic
[beta_sample]=BQR(Y,X,Q);
toc

beta_mean=mean(beta_sample,2);
L2=norm(beta_mean-BetaTrue);





