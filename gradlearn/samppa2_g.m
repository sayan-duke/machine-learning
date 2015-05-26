function [pa2]=samppa2_g(pa, pa2, hypa, rc);
% Sample the pa2 parameters.

if(rc=='c')
    m=length(pa.be); 
    if (length(pa2.T)==1)
        T=1./gamrnd((hypa.atau+m)/2,1./((sum(pa.be.^2)+hypa.btau)/2),1,1);
    else
        T=1./gamrnd((hypa.atau+1)/2,1./((pa.be.^2+hypa.btau)/2),m,1);
    end
    pa2.T=T; 
end

n=size(pa.C,2);
pa2.phic=gamrnd((hypa.aphic+n)/2, 1./((hypa.bphic+sum(pa.C.^2,2))/2));
