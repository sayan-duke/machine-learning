function [C]=sampC_g(Y, K, pa, pa2, W, Fb, D, DW, DWD);

[n,ps]=size(D{1});   % depending on proj=T or F, the col# of D{1} can be s or p
tmp3=repmat(Y',n,1)-repmat(Fb,1,n);
diagphic=diag(pa2.phic);

%%% Now sequentially update c1,...cn. 
for j=1:n

    invV=zeros(ps); mu=zeros(ps,1);
    for i=1:n  % for vector c_j
        invV=invV+K(i,j)^2.*DWD{i};
        b=tmp3(:,i)-D{i}*(pa.C*K(:,i)-K(j,i).*pa.C(:,j));
        mu=mu+K(i,j).*DW{i}*b;
    end
    invV=pa.phi*invV; V=pinv(invV);
    mu=pa.phi*V*mu;     % likelihood for c_j: normal with par mu and V.


    %%% place a prior on the Ctilde (size: s*n) and sample from posterior
    postV=pinv(invV+diagphic);
    postmu=pinv(eye(ps)+V*diagphic)*mu;
    % pa.C(:,j)=mvnrnd(postmu', postV)';
    pa.C(:,j)=chol(postV)'*randn(ps,1)+postmu;


end

C=pa.C;