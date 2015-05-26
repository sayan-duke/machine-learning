function gamlr=mtngam(mu, Sig, gamlr_old)

gamlr=gamlr_old;

k=length(mu);

if k>1
    
    mkind=1:(k-1); 

    tmp=Sig(k,mkind)*(Sig(mkind,mkind)\eye(k-1));
    sig_kmk=sqrt(Sig(k,k)-tmp*Sig(mkind,k)); mu_kmk=mu(k)+tmp*(gamlr(mkind)'-mu(mkind));
    U=rand(1);
    gamlr(k)=mu_kmk+sig_kmk*norminv(U+(1-U)*normcdf(-mu_kmk/sig_kmk));
    if(gamlr(k)==Inf) gamlr(k)=0.1; end

    Sig_mkk=Sig(mkind,mkind)-Sig(mkind,k)*(1/Sig(k,k))*Sig(k,mkind);
    mu_mkk=mu(mkind)+Sig(mkind,k)*(1/Sig(k,k))*(gamlr(k)-mu(k));
    gamlr(1:k-1)=(chol(Sig_mkk)'*randn(k-1,1)+mu_mkk)';
    
else
    U=rand(1);
    gamlr(1)=mu+sqrt(Sig)*norminv(U+(1-U)*normcdf(-mu/sqrt(Sig)));
    if(gamlr(1)==Inf) gamlr(1)=0.1; end
    
end


