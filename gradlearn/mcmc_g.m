initialise_g;   % procedure for initialization 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start MCMC Analysis
A=zeros(n); B=zeros(n);
for it=1:(nmc+nit)
    if (it/500==floor(it/500)), fprintf('MCMC iteration %d\n',it); end

    %%% MCMC scheme
    if(rc=='c')
        for i=1:n
            A(:,i)=ones(n,1).*Z(i)-tmpD{i}*pa.C*K(:,i);   % A is cbind(a_i), where a_i=z_i*1-(1*x_i'-E)*C*K_i
        end
        % alpha0=sampintcpt_g(Y, A, Ker.F*pa.be, pa.phi, W);
        [pa.be]=sampBeta_g(A-alpha0, Ker.F(:,1:length(Ker.D)), pa2.T, W, hypa);  
        [pa.C]=sampC_g(Z-alpha0, K, pa, pa2, W, Ker.F*pa.be, tmpD, tmpDW, tmpDWD);
        [Z, postp]=sampleZ(alpha0+Ker.F(:,1:m)*pa.be(1:m), Y, n);  
        % Z(Z<0&Z>-5)=-5; Z(Z>0&Z<5)=5; 
    else 
        [pa.C]=sampC_g(Y-alpha0, K, pa, pa2, W, Y, tmpD, tmpDW, tmpDWD);
        [pa.phi]=sampphi_g(Y-alpha0, K, pa, W, tmpD);
    end

    [pa2]=samppa2_g(pa, pa2, hypa, rc);   

    savepost_g;   % save the posterior draws
end

