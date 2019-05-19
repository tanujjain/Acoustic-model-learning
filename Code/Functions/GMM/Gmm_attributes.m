function [ Pi_Gmm,mu,cov] = Gmm_attributes(X,Nmix_gmm)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

mu = zeros(size(X,1),Nmix_gmm);
cov = zeros(size(X,1),Nmix_gmm);
Pi_Gmm = zeros(Nmix_gmm,1);

    nsample = size(X,2);
              
        l=floor(nsample/Nmix_gmm);                
        beg=0;        
        
        for st = 1:Nmix_gmm
            beg=beg+1;
            last = st*l;
            Pi_Gmm(st,1) = l/nsample; 
            
            if st == Nmix_gmm
                last= last + mod(nsample,Nmix_gmm);  
                Pi_Gmm(st,1) = (l+mod(nsample,Nmix_gmm))/nsample;
            end
            
            mu(:,st)= sum(X(:,beg:last),2)/size(X(:,beg:last),2);
            cov(:,st) = sum((X(:,beg:last)-repmat(mu(:,st),1,size(X(:,beg:last),2))).^2,2)/size(X(:,beg:last),2);
            
            beg=last;                       
        end
       
end

