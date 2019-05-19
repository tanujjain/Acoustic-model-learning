% Script for executing Baum Welch algorithm for tuning parameters

clear;

%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/../../Functions/BW']);
addpath([pwd '/Cluster_Purity']);

%% Load Data

load X_48phoneme.mat;

NHmm = 48;
Pi = [1 0 0];
Nmix_gmm = 1;
Nstates = 3;

% Flat start initialization of the attributes
[Aest,mu,cov,globmu,globcov,Pi_Gmm,~] = flat_HMM_rand(X,NHmm,Nstates,Nmix_gmm);

% Call Sampler
[Aest,mu,cov,Pi_Gmm,SeqHMMLabel,HMMRemoved] = Baum_Welch(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm);
