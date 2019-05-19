function [mu,gamsum] = Getmean(X,Gamma)
% Function to update mean of every mixture of a state for a HMM 
% Gets the product of each sample in sequence X with the coresponding gamma (numerator in the update equation)
% Also gets the denominator of the update equation which is the sum of the gammas
% where:
% X: Sequence of samples (a single sequence)
% Gamma: probability of samples in X of being in a particular mixture of a given state and HMM
% return: mu - numerator in the udpate equation of mean which is implemented in the calling function
%       : gamsum - denominator in the update equation

    Nstates = size(Gamma,1);
    Nmix = size(Gamma,3);
    ObsDim = size(X,1);
    gamsum = zeros(Nstates,Nmix); % sum of gammas, denominator in the uddate equation implemented in the calling function
    
    mu = zeros(ObsDim,Nmix,Nstates);
    
    for i = 1:Nstates
        for j = 1:Nmix
            mu(:,j,i) = sum(repmat(Gamma(i,:,j),ObsDim,1).*X,2); % X(t)*gamma(t,i,j)
            gamsum(i,j) = sum(Gamma(i,:,j),2); % summation of gammas
        end
    end
    
end