function [B]=sampBeta_g(A, F, T, W, hypa)

[n,m]=size(F); 

if (length(T)==1) tau=T.*ones(m,1); 
else tau=T(1:m);
end 

V=pinv(F'*diag(sum(W,2))*F+diag(1./tau));
be_hat=V*F'*sum(W.*A,2);

% B=mvnrnd(be_hat', V)'; 
B=chol(V)'*randn(m,1)+be_hat; 
if (hypa.m>m) B=[B; zeros((hypa.m-m),1)]; end