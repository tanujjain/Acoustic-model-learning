function [cov] = Getcovariance(X,Gamma,mu)
% Function to update covariance of every mixture of a state for a HMM 
% Gets the product of each sample in sequence X with the coresponding gamma (numerator in the update equation)
% where:
% X: Sequence of samples (a single sequence)
% mu: mean of each gaussian mixture component for a given state and HMM
% Gamma: probability of samples in X of being in a particular mixture of a given state and HMM
% return: cov - numerator in the udpate equation of covariance which is implemented in the calling function


    Nstates = size(Gamma,1);
    Nsamples = size(Gamma,2);
    Nmix = size(Gamma,3);
    ObsDim = size(X,1);
    
    cov = zeros(ObsDim,Nmix,Nstates);
    
    for i = 1:Nstates
        for j = 1:Nmix
            cov(:,j,i) = sum(repmat(Gamma(i,:,j),ObsDim,1).*(X - repmat(mu(:,j,i),1,Nsamples)).^2,2); % (X(t)-mu(i,j))^2*gamma(t,i,j)
        end
    end
    
end