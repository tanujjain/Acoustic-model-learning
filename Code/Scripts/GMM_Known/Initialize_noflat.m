function [Aest,mu,cov,Pi_Gmm] = Initialize_noflat(NHmm,Nmix,Nstates,GlobalMean,GlobalCov)
% Function to implement random start initialization
% where:
%   NHmm: number of HMMs
%   Nmix: number of Mixtures per GMM
%   Nstates: number of states
%   GlobalMean: global mean of all samples. (ObsDim*1) vector
%   GlobalCov: Global covariance of all samples. (ObsDim*1) vector
% return:
%       Aest: Transition matrices. (NHmm*1) cell with (Nstates*Nstates) matrix for Hmms with Nstates number of states
%       mu: NHmm by 1 cell, contains means, each cell contains (ObsDim*Nmix*Nstates) matrix
%       cov: NHmm by 1 cell, contains covariances, each cell contains (ObsDim*Nmix*Nstates) matrix
%       Pi_Gmm: NHmm by 1 cell, contains mixture weights, each cell contains (Nmix*Nstates) matrix

Aest = cell(1);
mu =  cell(1);
cov =  cell(1);
Pi_Gmm =  cell(1);

Dir_Aest = ones(1,Nstates-1); % Dirichlet hyperparameters for transition matrices and mixture probabilities
Dir_PiGmm = ones(1,Nmix);

ObsDim = 39;

for hmm =1:NHmm
    Aest{hmm,:}=[dirichlet_gen(Dir_Aest) 0;0 dirichlet_gen(Dir_Aest);0 0 1];  % sample transition probabilities from Dirichlet distribution
    Pi = zeros(Nmix,Nstates); % equivalent to Pi_Gmm, used as a temporary variable
    muN = zeros(ObsDim,Nmix,Nstates); 
    covN = zeros(ObsDim,Nmix,Nstates); 
    Lambda = zeros(ObsDim,Nmix,Nstates); % precision
    
    for st = 1: Nstates
        Pi(:,st) = dirichlet_gen(Dir_PiGmm)'; % sample mixture weights from a dirichlet distribution
        for m = 1: Nmix
           Lambda(:,m,st) = gamrnd(1,(1./GlobalCov)); % sample precision from gamma distribution with scale=1, rate=1/global_covariance
           covN(:,m,st) = 1./Lambda(:,m,st); % covariance is the inverse of precision
           muN(:,m,st) = gaussgen(GlobalMean,diag(1./Lambda(:,m,st)),1);  % sample eman from gaussian distribution with mean as globalmean and variance (1/precision)          
        end
        
    end
    Pi_Gmm{hmm,:} = Pi; 
    mu{hmm,:} = muN;
    cov{hmm,:} = covN;   
end
