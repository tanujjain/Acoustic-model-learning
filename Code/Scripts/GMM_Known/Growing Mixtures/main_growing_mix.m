clear;
% Main script for known HMM number case
% for growing mixtures
%% Add paths for all functions

addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Database');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Functions/Common_Functions');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Functions/GMM');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Scripts/GMM_Known/Growing Mixtures/Cluster_Purity');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Scripts/GMM_Known');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Functions/GrowMix');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Functions/BW');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Scripts/BaumWelch');
addpath('/home/jain/Thesis/Acoustic_Model_Learning/Code/Functions/GrowMix');

%% Load Data
load X_48phoneme.mat;

NHmm = 48;
Pi = [1 0 0];
Nmix = 1;
Nstates = 3;
[Aest,mu,cov,~,~,Pi_Gmm,~] = flat_HMM_rand(X,NHmm,Nstates,Nmix);  % Flat start initialization

load GlobMeanCov_48.mat;

%% Sampler

[Aest,mu,cov,SeqHMMLabel,HMMRemoved] = Multi_HMM_Sampler_GrowingMix(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm,globmu,globcov);

