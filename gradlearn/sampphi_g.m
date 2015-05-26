function [phi]=sampphi_g(Y,K,pa,W,D)

n=size(D{1},1); 

resid=zeros(n,n);

for i=1:n
    resid(:,i)=ones(n,1).*Y(i)-Y-D{i}*pa.C*K(:,i);
end

resid=sum(sum(W'.*(resid.^2)));
phi=gamrnd(n^2/2, 1/(resid/2),1,1);
