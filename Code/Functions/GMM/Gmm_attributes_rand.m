function [ Pi_Gmm,mu,cov ] = Gmm_attributes_rand( X,Nmix_gmm )
% Function for flat start initialization of GMM for a given state and HMM
% where:
%       X:  All samples assigned to this state of this HMM. (ObsDim*num_samples) matrix
%       Nmix_gmm: number of mixtures per GMM
% return:
%       Pi_Gmm: Mixture weights for the GMM. (Nmix_gmm*1) vector
%       mu: means for all mixtures for this state. (ObsDim*Nmix_gmm) matrix
%       cov: covariances for all mixtures for this state. (ObsDim*Nmix_gmm) matrix

mu = zeros(size(X,1),Nmix_gmm);
cov = zeros(size(X,1),Nmix_gmm);
Pi_Gmm = zeros(Nmix_gmm,1);

mixlabel= zeros(1,size(X,2));
mixProbs = (1/Nmix_gmm)*ones(1,Nmix_gmm);  % Assign equal probability to each mixture

for k=1:size(X,2)
    mixlabel(k) = AssignLabel(mixProbs);  % Assigning initial labels to each sample
end

for m = 1:Nmix_gmm
    mu(:,m) = mean(X(:,mixlabel == m),2);
    cov(:,m) = sum((X(:,mixlabel == m)-repmat(mu(:,m),1,size(X(:,mixlabel == m),2))).^2,2)/(size(X(:,mixlabel == m),2)-1);
    Pi_Gmm(m) = size(X(:,mixlabel == m),2)/size(X,2);
end

end

