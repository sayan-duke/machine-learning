% Save the posterior draws and calculate the posterior probabilities

if (it>nit)

    C=VM*pa.C;  
    if(rc=='c') mpostp = mpostp+postp; end  % will be divided by the # of chain to get the means later     

    for k=1:p
        tempgn(k)=sqrt(C(k,:)*K*C(k,:)');
    end
    nm.gnorm=[nm.gnorm tempgn];
    
    if nargout>2 nm.gop=[nm.gop reshape(C*K*K*C',p^2,1)]; end
    
    if nargout>1
        [gu,gd,gv]=svd(C*K,0);
        if it==nit+1
            gd=diag(gd); cgd=cumsum(gd);
            ednum=min(find(cgd/cgd(end)>0.9, 1),5);
            for k=1:ednum nm.V{k}=[]; end
            gucol=size(gu,2);
            ref=ones(gucol,1);
            absgu=abs(gu);
            for j=1:gucol ref(j)=find(absgu(:,j)==max(absgu(:,j))); end
        end
        gu=gu*diag(2*(diag(gu(ref,1:gucol))>0)-1);
        for k=1:ednum nm.V{k}=[nm.V{k} gu(:,k)]; end
    end
end