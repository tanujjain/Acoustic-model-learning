% Script to implement unsegmented speech recordings with 
% unknown number of HMMs

clear;

%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/../../Functions/GMM_Unknown']);
addpath([pwd '/Segmentation']);
addpath([pwd '/../Unknown_number']);
addpath([pwd '/Cluster_Purity']);

%% Load Data
load X_Unsegmented.mat;
load AbsAudioBound.mat;
load Unseg_SeqOrigLabel.mat;

% NHmm = 48;
Pi = [1 0 0];
Nmix = 8;
Nstates = 3;

load GlobMeanCov_48.mat;

%% Sampler

[Aest,mu,cov,SeqHMMLabel,HMMRemoved] = Multi_HMM_Sampler_Unsegmented(X_unsegmented,AbsAudioBoundaryIndApp,Nmix,globmu,globcov,Nstates,SeqOrigLab_Unseg);