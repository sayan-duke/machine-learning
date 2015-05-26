function Y=gramsmdt(mat);

[m,n]=size(mat);

Y=mat; Y(:,1)=Y(:,1)/norm(Y(:,1));

if(n>=2)
    
    for j=2:n       
        Y(:,j)=mat(:,j)-Y(:,1:j-1)*Y(:,1:j-1)'*mat(:,j);
        Y(:,j)=Y(:,j)/norm(Y(:,j));
    end
    
end

