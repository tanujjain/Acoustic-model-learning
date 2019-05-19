function [mu,cov,Pi_Gmm] = EM(X,mu,cov,Pi_Gmm,nclass)
% Function to tune the attributes of a GMM as a part of the process of splitting the mixture components
% where:
%   X: All samples for a given state and HMM. (ObsDim*sample_count) matrix
%   mu: means of all mixture components for this state and HMM. (ObsDim*Nmix) matrix
%   cov: covariances of all mixture components for this state and HMM. (ObsDim*Nmix) matrix
%   Pi_Gmm: mixture weights of all mixture components for this state and HMM. (Nmix*1) vector
%   nclass: number of mixture components

ObsDim = size(X,1);
for iter = 1:10 
Posterior = zeros(nclass,size(X,2)); % posterior probability of each sample to belong to each class
    
    for n = 1:size(X,2)
        for k = 1:nclass
        Posterior(k,n) = log(Pi_Gmm(k)) + LogLikelihood(X(:,n),mu(:,k),cov(:,k));
        end
        Posterior(:,n) = exp(Posterior(:,n) - logaddsum(Posterior(:,n),nclass)); % normalization
    end
    
%% Update mu,cov,Pi_Gmm
    muout = (Posterior*X')./repmat(sum(Posterior,2),1,size(mu,1));
    mu = muout';
    Pi_Gmm = sum(Posterior,2)/size(X,2);

    for k = 1:nclass
        cov(:,k) = sum(repmat(Posterior(k,:),size(X,1),1).*(X - repmat(mu(:,k),1,size(X,2))).^2,2)/sum(Posterior(k,:),2);
        for i = 1:ObsDim
            if cov(i,k)<0.01
            cov(i,k) = 0.01; % Variance flooring for each dimension
            end
        end
    end
end

end