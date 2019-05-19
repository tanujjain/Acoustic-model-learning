clear;
% Main script for known HMM number case
% with 61 phonemes
%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/Cluster_Purity']);

%% Load Data

load X_min_seq_3_no_overlap.mat;  % For initial set of 61 phonemes
load Init_Gmm_noflat.mat; % random intialised attributes
load GlobMeanCov.mat;
NHmm = 61;
Pi = [1 0 0];

%% Sampler

[Aest,mu,cov,SeqHMMLabel,HMMRemoved] = Multi_HMM_Sampler_Pruned_N(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm,GlobalMean,GlobalCov);

