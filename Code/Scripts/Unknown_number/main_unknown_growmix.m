% Script for unknown number of HMM case with growing number of GMM components

clear;

%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/../../Functions/GMM_Unknown']);
addpath([pwd '/../../Functions/GrowMix']);
addpath([pwd '/Cluster_Purity']);

%% Load Data

load X_48phoneme.mat;
load GlobMeanCov_48.mat;

%% Attributes
Nmix = 1;
Nstates = 3;

%% Call Sampler

[AestOut,muOut,covOut,Pi_GmmOut,SeqPerHMMOut] = HMM_Sample_Unknown_growmix(X,Nmix,globmu,globcov,Nstates,SeqOrigLabel);
