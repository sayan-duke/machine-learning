function [Ker]=myK_eigen_g(K, m);
% Decompose the kernel matrix
% returning:
% Ker.F : nxm eigen-matrix 
% Ker.D : nxm1 digonal (it's possible m1<m)
% Ker.mean: row mean of K

epsilon=1e-3;
[n,N]=size(K);

Ker.mean = mean(K,1); 
[U D V] = svd(K); D=diag(D); U=U(:,1:m); % SVD decomposition
m_old = m;
m=min(m_old,length(find(D>epsilon)));
D=D(1:m); 
if (m<m_old) U(:,((m+1):m_old))=zeros(n,(m_old-m)); end
Ker.F=U*diag([D;zeros(m_old-m,1)]); Ker.D=D; 

% if it turns out that the actual effective dimension # is even less than
% m, then add up 0 columns to U, but length(D) is still the # of actual
% effective dimensions. 
