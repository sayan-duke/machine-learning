% function [Bom, Bpost]=bmi(Y, X, d, res, pre_pca, niter)
%  bmi: Bayesian mixture inverse regression for supervised dimension reduction.  
%  Return the e.d.r. directions for the posterior mean and posterior draw. See below for details. 
%
% Input: Y is the response, a n-by-1 vector with n the sample size. 
%        X is the covariates matrix (n-by-p), with rows corresponding to observations and columns to covariates.
%        d: number of e.d.r. directions, or equivalently the dimension of the e.d.r. subspace. 
%        res: a structure that related to the response: 
%             res.type='c': the response is continuous; in this case
%               res.ker_bandwidth specifies the kernel bandwidth parameter 
%                (which is the reciprocal of the square root of the kernel precison parameter). Default bandwidth = 0.8.
%               res.num_loc: number of locations/atoms in the kernel stick breaking prior distribution.
%             res.type='d': the response is discrete; in this case
%               res.alpha0 specifies the concentration parameter for the base measure. Default = 1. 
%        pre_pca: 'T' or 'F': pre-process by principle component analysis or not? Useful for high dimensional problems; Default = 'F'.
%        niter=[number of burn-in iterations; number of posterior draws]; Default = [1000;1000].
%
% Output: 
%        Bom: the orthonormalized posterior mean for the e.d.r. This is a p-by-d matrix with columns spanning the e.d.r. subspace. 
%        Bpost: the posterior draws for the e.d.r. This is a p*d-by-(number of posterior draws) matrix. 

tic

% if nargin<2 display('Must input both the response and covariates!'); return; end
% if nargin<3 display('Must input the number of e.d.r. directions!'); return; end
% if(nargin<4 || isfield(res,'type')==0) display('Must input the response type!'); return; end

[n,p]=size(X); 

% if nargin<6 niter=[1000; 1000]; end
% if nargin<5 pre_pca='F'; end
%if(res.type=='c' && isfield(res,'ker_bandwidth')==0) res.ker_bandwidth=0.8; end
%if(res.type=='c' && isfield(res,'num_loc')==0) res.num_loc=n/4; end
%if(res.type=='d' && isfield(res,'alpha0')==0) res.alpha0=1; end

pa2.ah=1; pa2.bh=1; % control clus.V

X=(X-repmat(mean(X),n,1)); %./repmat(std(X),n,1); 
if (res.type=='c') Y=Y./std(Y); end
Z=X'; %clear X; 

if (p>n) pre_pca='T'; end
if pre_pca=='T' % if pre-process by PCA
    [Vz Dz Uz]=svd(Z); Dz=diag(Dz);
    pst=find(cumsum(Dz.^2)/sum(Dz.^2)>0.8,1); % select the first components that explain 80% variation.
    Vz=Vz(:,1:pst); Dz=Dz(1:pst); Z=diag(Dz)*Uz(:,1:pst)'; clear Uz;
else pst=p; 
end

nit=niter(1); nmc=niter(2); res_type=res.type; 
if(res_type=='c') ker_bandwidth=res.ker_bandwidth; num_loc=res.num_loc; 
else alpha0=res.alpha0; 
end

%%% Initialize
gam=zeros(pst,d); nu=zeros(d,n);
Dinv=eye(pst); eyemat=eye(pst);

%%% Hyper-parameters
pa.phig=0.01; pa2.ga=0.01; pa2.gb=0.01; % hyper for gam
pa2.Ddf=pst+1; pa2.Dpc=0.01*eye(pst);

Bpost=[]; % This will be the posterior draws for the e.d.r. directions. 

if res_type=='c' % if the response is continuous
    pre=1/ker_bandwidth^2; % set the kernel precison parameter
    clus.lab=ones(n,1); clus.atom=zeros(d,num_loc); % cluster labels and atoms. 
    clus.V=[zeros(1,num_loc-1) 1]; clus.A=-1*ones(n,num_loc); clus.B=-1*ones(n,num_loc); 
    clus.num=zeros(1,num_loc); % how many samples belong to each cluster
    loc_min=min(Y)-range(Y)/num_loc; loc_max=max(Y)+range(Y)/num_loc;
    clus.loc=random('unif',loc_min,loc_max, 1, num_loc); % cluster locations
    ker=zeros(n,num_loc); for h=1:num_loc ker(:,h)=KerYL(Y,clus.loc(h),pre); end % calculate the kernel (between Y and locations). 
else % if the response is discrete
    alpha0=1; % precision parameter for the base measure
    uY=unique(Y); uY=sort(uY); H=length(uY); % H is the number of classes. 
    clus.num=ones(H,1); % how many clusters are there in each class
    clus.lab=ones(2,n); % the first row specifies each sample belongs to which class
                        % the second row specifies each sample belongs to which cluster
    clus.nu=cell(H,1);  % clus.nu{h} is a d-by-clus.num(h) matrix for the atoms in class h.  
    clus.mem=cell(H,1); % clus.mem{h} for class h is a vector of dimension clus.num(h) that specifies the number of samples that belong to each cluster. 
    for h=1:H 
        clus.nu{h}=zeros(d,1); 
        indh=find(Y==uY(h));
        clus.lab(1,indh)=h; clus.mem{h}=[length(indh)];
    end
end



for t=1:(nit+nmc) % begin MCMC update
    
%%%%% Sample Gamma

    for l=pst:-1:1 % update sequentially for each row of gam

        if(l<pst) 
            Z_nl=Z_nl+(eyemat(:,l)*gam(l,:)-eyemat(:,l+1)*gam(l+1,:))*nu;
        else Z_nl=Z-(gam-[zeros(pst-1,d); gam(pst,:)])*nu;
        end

        dl=min(d,l);
        Sig_gam=(Dinv(l,l)*(nu(1:dl,:)*nu(1:dl,:)')+pa.phig*eye(dl))\eye(dl);
        mu_gam=Sig_gam*(nu(1:dl,:)*Z_nl')*Dinv(:,l);

        if(l>d) % update the l-th row. 
            gam(l,1:dl)=(chol(Sig_gam)'*randn(dl,1)+mu_gam)';
        else % for diagonal elements sample from a truncated normal. 
            gam(l,1:dl)=(chol(Sig_gam)'*randn(dl,1)+mu_gam)'; 
            if(gam(l,dl)<0) gam(l,1:dl)=mtngam(mu_gam,Sig_gam,gam(l,1:dl)); end
        end
    end
    
%%%%% 
    
%%%%% Sample nu: procedure differs for different response type. 

if res_type=='d' % if response discrete
    
    %%% some intermediate quantities that remain constant in each main MCMC iteration. 
    Dg=Dinv*gam; gDgI=gam'*Dg+eye(d); gDgIinv=pinv(gDgI); chol_g=chol(gDgIinv)'; det_g=det(gDgI)^(-0.5);
    ggD=gDgIinv*Dg'; tmpDD=Dinv-Dg*ggD; 
    q0=alpha0*det_g*exp(-0.5*sum(Z.*(tmpDD*Z))); q0(q0<1e-10)=1e-10;
    
    for i=1:n
        
        h0=clus.lab(1,i); c0=clus.lab(2,i);
        if clus.mem{h0}(c0)==1 % if sample i is the only one in its cluster
            clus.num(h0)=clus.num(h0)-1; clus.nu{h0}(:,c0)=[]; clus.mem{h0}(c0)=[];
            clus.lab(2, (clus.lab(1,:)==h0) & (clus.lab(2,:)>c0))=clus.lab(2, (clus.lab(1,:)==h0) & (clus.lab(2,:)>c0))-1; 
            % cluster labels for those with the same class but larger cluster labels as sample i should be reduced by 1. 
        else clus.mem{h0}(c0)=clus.mem{h0}(c0)-1; % else only the number of samples in that cluster is reduced by 1. 
        end
        
        %%% update the cluster label for sample i
        Zgu=Z(:,i)*ones(1,clus.num(h0))-gam*clus.nu{h0};
        qclus=exp(-0.5*sum(Zgu.*(Dinv*Zgu))); qclus(qclus<1e-10)=1e-10; % the likelihood contribution    
        prob=[q0(i) qclus.*clus.mem{h0}]; prob=prob./sum(prob); % combine the likelihood contribution and the prior contribution. 
        whclus=rand_mn(prob); whclus=find(whclus==1); 
        
        if(whclus==1) % if should create a new cluster
            nu(:,i)=chol_g*randn(d,1)+ggD*Z(:,i);
            clus.num(h0)=clus.num(h0)+1; clus.nu{h0}=[clus.nu{h0} nu(:,i)]; clus.mem{h0}=[clus.mem{h0} 1]; clus.lab(2,i)=clus.num(h0); 
        else whclus=whclus-1; nu(:,i)=clus.nu{h0}(:,whclus); clus.mem{h0}(whclus)=clus.mem{h0}(whclus)+1; clus.lab(2,i)=whclus; 
        end
        
    end
%%%%%  

else % if response continuous

    %%% sample cluster labels
    
    gca=gam*clus.atom;
    for i=1:n
        
        residi=Z(:,i)*ones(1,num_loc)-gca;
        liki=exp(-0.5*sum(residi.*(Dinv*residi))); % likelihood contribution
        ui=clus.V.*ker(i,:); ui(num_loc)=1; uic=cumprod(1-ui); uic=[1 uic(1:num_loc-1)];
        pii=ui.*uic; pii=pii.*liki; pii=pii./sum(pii); % prior contribution
        
        whclusi=rand_mn(pii); whclusi=find(whclusi==1); 
        if(whclusi<clus.lab(i)) clus.A(i,whclusi+1:clus.lab(i))=-1; clus.B(i,whclusi+1:clus.lab(i))=-1; end  % "discard" those for h>K_i
        clus.lab(i)=whclusi; clus.A(i,whclusi)=1; clus.B(i,whclusi)=1; 
        
    end
    
    %%% sample atoms
    
    gD=gam'*Dinv; gDg=gD*gam;     
    for h=1:num_loc        
        clus.num(h)=sum(clus.lab==h);
        if(clus.num(h)>0)         
            varh=pinv(clus.num(h)*gDg+eye(d));
            meanh=varh*gD*sum(Z(:,clus.lab==h),2);
            clus.atom(:,h)=chol(varh)'*randn(d,1)+meanh;            
        else clus.atom(:,h)=randn(d,1); 
        end               
    end
    
    for i=1:n nu(:,i)=clus.atom(:,clus.lab(i)); end
    
    %%% sample V
    
    clus.V=random('beta', sum(clus.A.*(clus.A==1))+pa2.ah, sum((1-clus.A).*(clus.A==0))+pa2.bh); clus.V(num_loc)=1; 
    
    %%% sample A, B
    
    for i=1:n     
        if (clus.lab(i)>1)            
            p1=clus.V(1:clus.lab(i)-1).*(1-ker(i,1:clus.lab(i)-1));
            p2=(1-clus.V(1:clus.lab(i)-1)).*ker(i,1:clus.lab(i)-1);
            p3=(1-clus.V(1:clus.lab(i)-1)).*(1-ker(i,1:clus.lab(i)-1));
            probAB=[p1' p2' p3']; probAB=probAB./(sum(probAB,2)*ones(1,3));    
            
            whcombi=rand_mn(probAB); 
            clus.A(i,1:clus.lab(i)-1)=whcombi(:,1)'; clus.B(i,1:clus.lab(i)-1)=whcombi(:,2)'; 
        end
    end
    
    %%% sample locations   
    
    for h=1:num_loc
     
        if(sum(clus.B(:,h)==1)>0)
            Ysub=Y(clus.B(:,h)==1); loch_new=min(Ysub)+rand(1)*(max(Ysub)-min(Ysub));
        else 
            loch_new=Y(ceil(rand(1)*n));
        end

        ker_new=KerYL(Y,loch_new,pre);
        
        if (sum(clus.B(:,h))==-n)          % clus.B(:,h) all ==-1
            clus.loc(h)=loch_new; ker(:,h)=ker_new;
        else
            acpt=prod(ker_new(clus.B(:,h)==1)./ker(clus.B(:,h)==1),h);
            acpt=acpt*prod((1-ker_new(clus.B(:,h)==0))./(1-ker(clus.B(:,h)==0,h)));
            acpt=min(acpt,1);
            if(rand(1)<acpt) clus.loc(h)=loch_new; ker(:,h)=ker_new; end
        end        
     
    end
    
end
%%%%%  End: sample nu

%%%%% Sample Dinv
    resid=Z-gam*nu;
    Dinv=wishrnd((pa2.Dpc+resid*resid')\eye(pst),pa2.Ddf+n);  
%%%%%



    if(t>nit)
        if(pre_pca=='T')
             Bpost=[Bpost reshape(Vz*Dinv*gam,p*d,1)];
        else Bpost=[Bpost reshape(Dinv*gam,p*d,1)]; 
        end
        
    end
    
end

Bom=gramsmdt(reshape(mean(Bpost,2),p,d)); 

toc;




