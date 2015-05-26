%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Structures:
% (1) Ker (structure)
% F: nxn eigen matrix, the last (n-m) cols are zero
% D: nxm eigen values
% mean: 1xn mean of the kernel matrix
% (2) nm (structure used to store posterior draws)
% be, T
% (3) hypa (structure for hyper-parameters)
% atau, btau: Gamma(atau/2, btau) for coef precision T
% aphic, bphic; 
% m: # of factors
% (4) pa2 (structure for hierarchical (2nd layer) parameters
% T: nx1 variance for be
% phic
% (5) pa (structure for model parameters)
% be, C, phi

nm.gop=[];
tempgn=zeros(p,1); nm.gnorm=[];

% priors hyperparameters         
hypa.atau=2;  hypa.btau=2; 
hypa.aphic=2; hypa.bphic=2;

% starting values for all parameters
pa.rho=ones(p,1)./p; K=myK_str(X, X, pa.rho, Ker_type);
% Ker.mean = mean(K,1); 
% K = K - repmat(Ker.mean, n,1) -repmat(Ker.mean', 1, n) + mean(Ker.mean).*ones(n,n); % centering the kernel matrix 
% K = (K+K')./2;
pa2.phic=ones(s,1); 
pa.C=zeros(s,n);
alpha0=0;   % alpha0 is the intercept;
if(rc=='c')
    pa.be = zeros(hypa.m,1); mpostp=0;
    [Ker]=myK_eigen(K, hypa.m);  % calculate the kernel matrix for the training sample and do decomposition
    m=min(hypa.m,length(Ker.D));
    if mulT=='T' pa2.T=ones(hypa.m,1);
    else pa2.T=1;
    end
    [Z]=sampleZ(alpha0+Ker.F(:,1:m)*pa.be(1:m), Y, n); 
end
pa.phi=1;

