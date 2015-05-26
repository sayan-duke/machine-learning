function y=rand_mn(p)

[d1,d2]=size(p);
y=zeros(d1,d2);

for i=1:d1
    y(i,find(rand(1)<cumsum(p(i,:)),1))=1;
end