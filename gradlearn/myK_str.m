function [K]=myK_str(X, Xtest, rho, Ker_type);
% Calculate the kernel matrix in a straight way (no distance matrix needed)

if iscell(Ker_type)   d=Ker_type{2}; Ker_type=Ker_type{1};   end
[n,p]=size(X); ntest=size(Xtest,1); 

if strcmp(Ker_type,'linear')  K=Xtest*diag(rho)*X';
elseif strcmp(Ker_type,'Gaussian') 
    K=zeros(ntest,n);
    for i=1:ntest
        for j=1:n
            K(i,j)=(Xtest(i,:)-X(j,:)).^2*rho;
        end
    end
    K=exp(-K);
else K=Xtest*diag(rho)*X'; K=(1+K).^d;
end