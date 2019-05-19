% Function for testing supervised performance
% for fixed number of GMM components

clear;

%% Add paths for all functions

addpath([ pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
addpath([pwd '/../BaumWelch']);
addpath([pwd '/../../Functions/BW']);

% Load Training data results
load Training_Data_48.mat ;

% Load Validation data Sequences
load Validation_Data_48.mat;


load GlobMeanCov_48.mat;
NHmm = 48;
Pi = [1 0 0];
  
Chooselearningalgo = input(['\n Choose learning algorithm:','\n Enter 1 : Gibbs Sampling','\n       2 : Baum Welch algorithm \n'] );

switch Chooselearningalgo
    case 1       
        algotoken = 0;
    case 2
        algotoken = 1;
    otherwise
        disp('Option value not correct');
end

% Call function to classify
[AestN,muN,covN,Pi_GmmN] = Multi_HMM_Sampler_Test(XTraining,XValidate,Aest,mu,cov,NHmm,Pi,...
                                                            SeqOrigLabel_TrainingData,SeqOrigLabel_ValidationData,Pi_Gmm,globmu,globcov,algotoken);
                                                        