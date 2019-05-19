
function [Aest,mu,cov,Pi_Gmm] = Multi_HMM_Sampler_Test(XTrain,XValid,Aest,mu,cov,NHmm,Pi,...
                                                                SeqOrigLabel_TrainingData,SeqOrigLabel_ValidationData,Pi_Gmm,globmu,globcov,algotoken)
% Sampler function to test supervised performance for static number of mixture components
% where:                                                           
%   XTrain: training sequences. (No_of_training_sequences * 1) cell with training sequence in each cell
%   XValid: validation sequences. (No_of_validation_sequences * 1) cell with training sequence in each cell
%   Aest: initial transition matrices. (48*1) cell with (Nstates*Nstates) matrix for Hmms with Nstates number of states
%   mu: 48 by 1 cell, contains initial means for 48 phonemes (HMMs), each cell contains (ObsDim*Nmix*Nstates) matrix
%   cov: 48 by 1 cell, contains initial covariances for 48 phonemes (HMMs), each cell contains (ObsDim*Nmix*Nstates) matrix
%   Pi_Gmm: 48 by 1 cell, contains initial covariances for 48 phonemes (HMMs), each cell contains (Nmix*Nstates) matrix
%   SeqOrigLabel_TrainingData: (number_of_training_seq * 1) vector. Contains original sequence labels of training dataoriginal labels of the sequences
%   SeqOrigLabel_ValidationData: (number_of_validation_seq * 1) vector. Contains original sequence labels of validation sequences
%   algotoken: 0 for using Gibbs sampling, 1 for Baum Welch

%Attributes
Nstates = size(Aest{1,:},1);

%% Hyper Parameter Initialization

beta_Aest = ones(Nstates-1,Nstates-1,NHmm); % Dirichlet parameters for sampling Transition Matrix probabilities
SeqHMMLabel = SeqOrigLabel_TrainingData;

  
   %% Iterations
NumItr = 150;

for it = 1:NumItr
    tic
    sprintf('Iteration no = %d\n',it)
    
   
%% Sample parameters of individual HMMs

   parfor k=1:NHmm    
    
        disp(k);

        SameLabelSeq = XTrain(SeqHMMLabel == k);
        seqlen_new = cellfun(@(x) size(x,2), SameLabelSeq);
        
        if algotoken == 0
            [c1,c2,c3,c4] = BlockSampler(SameLabelSeq,seqlen_new,Aest{k,:},mu{k,:},cov{k,:},beta_Aest(:,:,k),Pi,Pi_Gmm{k,:},globmu,globcov);
        else
            [c1,c2,c3,c4] = BWSampler(XTrain(SeqHMMLabel == k),Aest{k,:},mu{k,:},cov{k,:},Pi,Pi_Gmm{k,:},k); % For Baum-Welch training
        end
        
        Aest{k,:} = c1;
        mu{k,:} = c2;
        cov{k,:} = c3;
        Pi_Gmm{k,:} = c4;
                 
   end
   disp('* * * Classifying Training Data * * *');
   ClassifyHMM(XTrain,Aest,mu,cov,NHmm,Pi,SeqOrigLabel_TrainingData,Pi_Gmm,0,it);
   disp('* * * Classifying Validation Data * * *');
   ClassifyHMM(XValid,Aest,mu,cov,NHmm,Pi,SeqOrigLabel_ValidationData,Pi_Gmm,1,it);
   
toc
end

end