function [Z, postp] = sampleZ(Zhat, Y, n); 
% Sample the latent var Z (n by 1)
% record postprob of being 1 for n obs (n by 1) in postp, and draw new Z
% Zhat is the predicted value by the model
eps=10e-10;
P=normcdf(-Zhat);
postp=ones(n,1)-P; 
postp(find(postp<eps))=eps; postp(find(postp>1-eps))=1-eps;

u=rand(n,1); 
invP=Y.*P + u.*(P+(1-2.*P).*Y);
invP(find(invP<eps))=eps; invP(find(invP>1-eps))=1-eps;
Z = Zhat + norminv(invP);