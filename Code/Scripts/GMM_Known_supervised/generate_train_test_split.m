% Function to generate training data and
% validation data

clear all;
close all;

addpath([pwd '/../../Database']);
addpath([pwd '/../../Functions/Common_Functions']);
addpath([pwd '/../../Functions/GMM']);
% Attributes
NHmm = 48; % Number of HMMs
Nstates = 3; % Number of States per HMM
Pi=[1 0 0];  % Left-Right HMM initial state probabilities
Nmix_gmm = 8;

%% Load Data

load X_48phoneme.mat;

SeqOrigLabel=SeqOrigLabel+1;
while(1)
TrainingDataIndex = randperm(size(X,1),floor(0.75*size(X,1))); % generate unique indices of sequences to be assigned as training
ValidationDataIndex =  setdiff(1:size(X,1),TrainingDataIndex); % use remaining sequences to be used as validation sequences

XTraining = X(TrainingDataIndex);
XValidate = X(ValidationDataIndex);
SeqOrigLabel_TrainingData = SeqOrigLabel(TrainingDataIndex); % Store original labels of the sequences
SeqOrigLabel_ValidationData = SeqOrigLabel(ValidationDataIndex);

    if(size(unique(SeqOrigLabel_TrainingData),1) == NHmm) % Ensure that sequences from all classes are represented in training data
        break;
    end
    
end
%% Choose initialization
  tic  
    ChooseInitialization = input([' Enter 1 : Flat start','\n       2 : k-means Initialization'...
       '\n       3 : k-means Initialization using sequence means'] );

switch ChooseInitialization
    case 1       
        [Aest,mu,cov,~,~,Pi_Gmm,~] = flat_HMM_rand(XTraining,NHmm,Nstates,Nmix_gmm); % flat start initialization
    case 2
        [Aest,mu,cov,globmu,globcov] = kmeans_init(X,NHmm,Nstates); % Untested, not safe to use anymore
    case 3       
        [Aest,mu,cov,globmu,globcov] = kmeans_means(X,NHmm,Nstates); % Untested, not safe to use anymore
    otherwise
        disp('Option value not correct');
end
toc
disp('Initialization ended');


%% Save training and test data and attributes

save('Training_Data_48.mat','XTraining','Aest','mu','cov','Pi_Gmm','TrainingDataIndex','SeqOrigLabel_TrainingData');
save('Validation_Data_48.mat','XValidate','SeqOrigLabel_ValidationData','ValidationDataIndex');

