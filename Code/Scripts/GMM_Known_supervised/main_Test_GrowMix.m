% Function for testing supervised performance
% for growing number of GMM components

clear;

%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/../BaumWelch']);
addpath([pwd '/../../Functions/BW']);
addpath([pwd '/../../Functions/GrowMix']);

% Load Training data results
load Training_Data_48.mat ;
clear mu; % clear these attributes as these may correspond to values for different mixture count
clear cov;
clear Aest;
clear Pi_Gmm;

NHmm = 48;
Pi = [1 0 0];
Nmix = 1;
Nstates = 3;
[Aest,mu,cov,~,~,Pi_Gmm,~] = flat_HMM_rand(XTraining,NHmm,Nstates,Nmix); % Flat start initilization using training sequecnes for Nmix = 1

% Load Validation data Sequences
load Validation_Data_48.mat;

load GlobMeanCov_48.mat;

Chooselearningalgo = input(['\n Choose learning algorithm:','\n Enter 1 : Gibbs Sampling','\n       2 : Baum Welch algorithm \n'] );

switch Chooselearningalgo
    case 1       
        algotoken = 0;
    case 2
        algotoken = 1;
    otherwise
        disp('Option value not correct');
end

%% Call function to classify
[AestN,muN,covN,Pi_GmmN] = Multi_HMM_Sampler_Test_GrowMix(XTraining,XValidate,Aest,mu,cov,NHmm,Pi,...
                                                            SeqOrigLabel_TrainingData,SeqOrigLabel_ValidationData,Pi_Gmm,globmu,globcov,algotoken);
                                                        