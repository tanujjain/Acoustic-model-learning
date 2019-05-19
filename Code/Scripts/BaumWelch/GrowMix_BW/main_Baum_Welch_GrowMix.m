
% Main script for known HMM number case
% with 48 phonemes to test growing mixture scenario with Baum-Welch algorithm
clear;

%% Add paths for all functions


addpath([ pwd '/../../../Database']);
addpath([pwd '/../../../Functions/Common_Functions']);
addpath([pwd '/../../../Functions/GMM']);
addpath([pwd '/../../../Functions/BW']);
addpath([pwd '/../../../Scripts/GMM_Known/Growing Mixtures']);
addpath([pwd '/../../../Scripts/BaumWelch']);
addpath([pwd '/Cluster_Purity']);

%% Load Data

load X_48phoneme.mat;

NHmm = 48;
Pi = [1 0 0];
Nmix_gmm = 1;
Nstates = 3;

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

%% Call Sampler
[Aest,mu,cov,Pi_Gmm,SeqHMMLabel,HMMRemoved] = Baum_Welch_GrowMix(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm);