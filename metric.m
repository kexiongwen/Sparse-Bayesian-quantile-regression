function[L2,L1,sparsity,Ham,FDR,FNDR,coverage,coverage_nonzero]=metric(beta_sample,betaTrue)

s=size(betaTrue);
nonzeros=(betaTrue~=0);
zeros=(betaTrue==0);
beta_mean=mean(beta_sample,2);
beta_std=std(beta_sample,[],2);


L2=norm(beta_mean-betaTrue);
L1=norm(beta_mean-betaTrue,1);
upp=quantile(beta_sample,0.95,2);
low=quantile(beta_sample,0.05,2);

nonzero_location=abs(beta_mean./beta_std)>1.96;
zero_location=ones(s(1))-nonzero_location;
sparsity=sum(nonzero_location);

coverage=sum((betaTrue>low) .* (betaTrue<upp))/s(1);
coverage_nonzero=sum((betaTrue(nonzeros)>low(nonzeros)) .* (betaTrue(nonzeros)<upp(nonzeros)))/sum(nonzeros);
Ham=sum(nonzeros~=nonzero_location);

FDR=(sum(nonzero_location)-sum(nonzeros.*nonzero_location))/sum(nonzero_location);
FNDR=(sum(zero_location)-sum(zeros.*zero_location))/sum(zero_location);

end