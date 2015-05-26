function [K]=KerYL(Y, L, pre);

ll=length(L);

if (ll==1) K=exp(-pre*(Y-L).^2); 
else K=exp(-pre*(Y*ones(1,ll)-ones(length(Y),1)*L).^2);
end