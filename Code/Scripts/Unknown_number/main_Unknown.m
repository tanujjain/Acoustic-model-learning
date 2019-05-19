% Function for unknown number of HMM case

clear;

%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/../../Functions/GMM_Unknown']);
addpath([pwd '/../GMM_Known']);
addpath([pwd '/Cluster_Purity']);

%% Load Data

load X_48phoneme.mat;
load GlobMeanCov_48.mat;

%% Attributes
Nmix = 8;
Nstates = 3;

%% Call Sampler

[AestOut,muOut,covOut,Pi_GmmOut,SeqPerHMMOut] = HMM_Sample_Unknown(X,Nmix,globmu,globcov,Nstates,SeqOrigLabel);