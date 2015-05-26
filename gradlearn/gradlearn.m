function [gnorm, dr, gop]=gradlearn(Y, X, rc, niter, Ker_type, st, hypam, hypas);
% GRADLEARN: Bayesian gradient learning for regression or binary classification. 
%  Return the RKHS norms for each dimension and (optionally) dimension reduction directions
%  and the GOP matrices, all of which are posterior draws.
%
% Input: Y is the response (if classification must be labeled as 1/0 or 1/-1), 
%         X is the covariates matrix, with rows corresponding to
%         observations and columns to covariates. 
%         Y and X are necessary inputs; 
%        rc=='r' does regression; rc=='c' does binary classification;
%         Default is 'r'
%        niter=[number of burn-in iterations; number of posterior draws];
%         Default is [1000;1000]
%        Ker_type=='Gaussian' for Gaussian kernel or 'linear' for Linear
%         kernel or {'poly',d} for polynomial kernel with order d; Default
%         is 'Gaussian'
%        st=='T'/'F' standardizes data first / or not. Default is 'T'
%        hypam: number of columns kept in F with K=FDF'; 
%        hypas: number of rows in \tilde{C}. 
%
% Output: 
%        gnorm: RKHS norms for the gradients. The t-th column is the t-th posterior draw. 
%        dr: dimension reduction directions. V{i} is a matrix for the i-th dimension
%         reduction direction of which the t-th column is the t-th posterior draw.
%        gop: GOP matrices. The t-th column is the t-th posterior draw but
%         in a vector form. To recover the matrix form one needs
%         "reshape(gop(:,t),p,p)" where p is the number of dimensions. 


[n,p] = size(X);              % X: the sample matrix with n observations, each is a p-vector.
Y(Y==-1)=0;

if nargin<7 hypam=floor(min(min(p,n)/3,30)); end
if nargin<6 st='T'; end
if nargin<5 Ker_type='Gaussian'; end
if nargin<4 niter=[1000; 1000]; end
if nargin<3 rc='r'; end
if nargin<2 fprintf('Must input both the response and covariates');  return; end

nit=niter(1); nmc=niter(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hypa.m=hypam;   % # of features
mulT='F';       % do the beta's have different variances?

%%% standardize the training and test samples.
if st=='T'
    Xmean = mean(X,1); Xstd = std(X); Xstd(Xstd==0)=1;
    X = (X-repmat(Xmean, n,1))./repmat(Xstd,n,1);
end

%%% calculate the weight matrix
aa = zeros(n,n);
for i=1:n
  for j=1:n
      aa(i,j) = norm(X(i,:)-X(j,:));  
   end
end 
sigma = median(median(aa));
W=myK_str(X,X,ones(p,1)/(2*sigma^2),'Gaussian'); % for regression weight=pa.phi*W
clear aa sigma; 

% constructs the matrix of differences between all points
perM=0.9;
M=X(1:(end-1),:)'-repmat(X(end,:)', 1, n-1);    % size: p*(n-1)
[VM,dM,UM]=svd(M); dM=diag(dM);
if nargin<8
    vals = cumsum(dM.^2);
    s=find(vals/vals(end)>perM, 1);
else s=hypas;
end
VM=VM(:,1:s); dM=dM(1:s); 
clear UM; 
    
tmpD=cell(n,1); tmpDW=cell(n,1); tmpDWD=cell(n,1); 
for i=1:n 
    tmpD{i}=ones(n,1)*X(i,:)-X;  
    tmpD{i}=tmpD{i}*VM;
    tmpDW{i}=tmpD{i}'*diag(W(i,:));
    tmpDWD{i}=tmpDW{i}*tmpD{i};  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% now start MCMC analysis ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mcmc_g;    % The main MCMC sampling procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if rc=='c'
    mpostp=mpostp./nmc; fprintf('The training error rate is %f\n', sum(abs((mpostp>0.5)-Y))./n);
end

gnorm=nm.gnorm; 
if(nargout>1) dr=nm.V; gop=nm.gop; end
