clear;
% Main script for known HMM number case
% with 48 phonemes

%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/Cluster_Purity']);

%% Load Data
load X_48phoneme.mat;

NHmm = 48;
Pi = [1 0 0];
Nmix = 1;
Nstates = 3;
load GlobMeanCov_48.mat;

%% Choose initialization
Chooseinit = input(['\n Choose initialization algorithm:','\n Enter 1 : Flat start','\n       2 : Random Start \n'] );

switch Chooseinit
    case 1       
        inittoken = 0;
    case 2
        inittoken = 1;
    otherwise
        disp('Option value not correct');
end

if inittoken == 0
    [Aest,mu,cov,~,~,Pi_Gmm,~] = flat_HMM_rand(X,NHmm,Nstates,Nmix);  % Flat start initialization
else
    [Aest,mu,cov,Pi_Gmm] = Initialize_noflat(NHmm,Nmix,Nstates,globmu,globcov); % random initialization
end

%% Sampler

[Aest,mu,cov,SeqHMMLabel,HMMRemoved] = Multi_HMM_Sampler_Pruned_N(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm,globmu,globcov);

