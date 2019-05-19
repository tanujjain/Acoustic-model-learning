function [muout,covout,Pi_Gmmout] = Split_State(X,mu,cov,Pi_Gmm)
% Function to split gaussian components of the GMM of a given state
% where:
%   X: All samples assigned to a particular state for an HMM. (ObsDim*sample_count) matrix
%   mu: means of each GMM compenent of the given state. (ObsDim*Nmix) matrix
%   cov: covariances of each GMM compenent of the given state. (ObsDim*Nmix) matrix
%   Pi_Gmm: mixture weights of each GMM component for the given state. (Nmix*1) matrix
% return:
%   muout: means of each GMM compenent of the given state. (ObsDim*Nmix*2) matrix
%   covout: covariances of each GMM compenent of the given state. (ObsDim*Nmix*2) matrix
%   Pi_Gmmout: mixture weights of each GMM component for the given state. (2*Nmix*1) matrix

means_minus_delta = mu-sqrt(cov); % move mean one standard deviation to left and right
means_plus_delta = mu+sqrt(cov);

k = 2*size(mu,2); % get the new count of components
    
 %% K-Means to get mixture label for each sample and means to pass onto EM
 
 optns = statset('MaxIter',500); % set a mximum of 500 iterations to achieve convergence
 [ClustLabels,Centroids] = kmeans(X',k,'start',[means_minus_delta means_plus_delta]','options',optns); % get cluster labels for each sample and centroids for each cluster
 Centroids = Centroids';
 
 ClustCov = zeros(size(mu,1),k);
 
 % Get covariance of each cluster using samples asisgned to that cluster
 for i = 1:k
     SameClust = X(:,ClustLabels' == i);
     ClustCov(:,i) = sum((SameClust-repmat(Centroids(:,i),1,size(SameClust,2))).^2,2)/size(SameClust,2);
     if ClustCov(:,i)<0.01
         ClustCov(:,i) = 0.01; % variance flooring
     end
 end
 
 Pi_Gmm = 0.5*repmat(Pi_Gmm,2,1); % divide weights equally between newly formed mixture components
 
 % Use EM algorithm to further tune the means, covairance and mixture weights
 [muout,covout,Pi_Gmmout] = EM(X,Centroids,ClustCov,Pi_Gmm,k);
 
end