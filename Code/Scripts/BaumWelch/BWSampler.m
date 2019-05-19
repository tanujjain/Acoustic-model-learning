
function [Aest,mu,cov,Pi_Gmm] = BWSampler(X,Aest,mu,cov,Pi,Pi_Gmm,hmmnum)
% Function to implement Baum Welch algorithm to update parameters of an HMM
% where:
%   X: All sequences  assigned to this HMM. (num_sequence*1) cell
%   Aest: initial transition matrices. (Nstates*Nstates) matrix for an Hmm with Nstates number of states
%   mu: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
%   cov: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
%   Pi_Gmm: (Nmix*Nstates) matrix containing means for this HMM
%   Pi: Initial state probability (same for all HMMs due to left-right assumption)
%   hmmnum: the HMM nuber which is being updated
% return: updated attributes
% Terminology (alpha,beta,xsi,gamma) borrowed from Rabiner's paper (title: Tutorial on HMM and selected applications)

Nstates = size(Pi,2);
Nmix = size(Pi_Gmm,1);
ObsDim = size(X{1,:},1);

% Calculate B: observation probability, diff: Log(b for a mixture)-sum(b for all mixtures): for all
% samples and state
[B,Diff] = cellfun(@(x) State_Obs_Prob_GMM_each(x,mu,cov,Nstates,Pi_Gmm),X,'UniformOutput',0);

% Calculate Alpha (forward variable)
Alpha = cellfun(@(x) ForwardAlgorithmLog(x(:,:,end),log(Aest),log(Pi)),B,'UniformOutput',0); 

% Calculate Beta (backward variable)
Beta = cellfun(@(x) BackwardAlgorithmLog(x(:,:,end),log(Aest)),B,'UniformOutput',0);

% Calculate Gamma (probabiltiy of being in a state at a given time)
LogGam = cellfun(@(x,y) x+y,Alpha,Beta,'UniformOutput',0); 
    for n=1:size(X,1)
        LogGam{n,:} = LogGam{n,:}-repmat(cellfun(@(x) logaddsum(x,3),num2cell(LogGam{n,:},1)),Nstates,1);
    end

Gam = cellfun(@(x) exp(x),LogGam,'UniformOutput',0);

% This gamma gives the probabiltity of observation orignating from a given mixture of a state
LogGamma = cellfun(@(x,y) repmat(x,1,1,Nmix)+y,LogGam,Diff,'UniformOutput',0);    
    
Gamma = cellfun(@(x) exp(x),LogGamma,'UniformOutput',0);

% Calculate XSI
BplusBeta = cellfun(@(x,y) x(:,2:end,end)+y(:,2:end),B,Beta,'UniformOutput',0);

LogXSI = cellfun(@(x,y) CalcXSI(x(:,1:end-1),y,log(Aest)),Alpha,BplusBeta,'UniformOutput',0);
        
XSI = cellfun(@(x) exp(x),LogXSI,'UniformOutput',0); 

%% Updates
mu = zeros(ObsDim,Nmix,Nstates);
Den = zeros(Nstates,Nmix);
cov = zeros(ObsDim,Nmix,Nstates);

 % Update mu
for n=1:size(X,1)    

[mu1,Deno1] = Getmean(X{n,:},Gamma{n,:});
mu = mu+mu1;
Den = Den+Deno1;
end

for st =1:Nstates
    mu(:,:,st) = mu(:,:,st)./repmat(Den(st,:),ObsDim,1);
end

% Update cov
for n=1:size(X,1)
    cov1 = Getcovariance(X{n,:},Gamma{n,:},mu);
  ss = sum(sum(sum(isnan(cov1))));
  if ss~=0
      disp(n-1);
  end 

  cov = cov+cov1;
end 

for st =1:Nstates
    cov(:,:,st) = cov(:,:,st)./repmat(Den(st,:),ObsDim,1);
end

for st=1:Nstates
cov(:,:,st) = Floor_cov(cov(:,:,st),Nmix,hmmnum); % Floor covariance if less than 0.01 for each dimension and state
end

% Update Pi_Gmm
sums = zeros(Nstates,1,Nmix);
for n=1:size(X,1)
    sums = sums + sum(Gamma{n,:},2);
end

sums = reshape(sums,Nstates,Nmix);

for st=1:Nstates
    sums(st,:) = sums(st,:)/sum(sums(st,:));
end

Pi_Gmm = sums';

% Update Aest
gamsum = zeros(Nstates,1);
sumxsi = zeros(Nstates,Nstates);

for n=1:size(X,1)
    redgam = Gam{n,:};
    gamsum = gamsum+sum(redgam(:,1:end-1),2);
    sumxsi = sumxsi+sum(XSI{n,:},3);
end

Aest = sumxsi./repmat(gamsum,1,Nstates);
end