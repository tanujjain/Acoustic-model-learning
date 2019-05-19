
function [Aest,mu,cov,Pi_Gmm,SeqHMMLabel,HMMRemoved] = Multi_HMM_Sampler_Pruned_N(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm,globmu,globcov)
% Function to implement mixed sampling (Gibbs sampling/Baum Welch)
% where:
%   X: All sequences. (Num_sequences*1) cell
%   NHmm: number of HMMs
%   Aest: initial transition matrices. (NHmm*1) cell with (Nstates*Nstates) matrix for Hmms with Nstates number of states
%   mu: NHmm by 1 cell, contains means, each cell contains (ObsDim*Nmix*Nstates) matrix
%   cov: NHmm by 1 cell, contains covariances, each cell contains (ObsDim*Nmix*Nstates) matrix
%   Pi_Gmm: NHmm by 1 cell, contains mixture weights, each cell contains (Nmix*Nstates) matrix
%   Pi: Initial state probability (same for all HMMs due to left-right assumption)
%   SeqOrigLabel: Original class labels of the sequences. (Num_sequences*1) vector
%   SeqHMMLabel: Assigned cluster labels of the sequences. (Num_sequences*1) vector
%   globmu, globcov: global mean, covariance. (ObsDim*1) vectors
%   return: updated attributes + cell containing a list indices of removed HMMs + SeqHMMLabel

%Attributes
Nstates = size(Aest{1,:},1);

%% Hyper Parameter Initialization

beta_hmm = 1.0; % Dirichlet parameter for choosing an HMM for a sequence, same for all HMMs

beta_Aest = ones(Nstates-1,Nstates-1,NHmm); % Dirichlet parameters for sampling Transition Matrix probabilities
   
HMMProbs = (1/NHmm)*ones(1,NHmm); % InitiaSeqHMMLabell probability kept uniform
SeqHMMLabel = zeros(size(X,1),1);

for k=1:size(X,1)
    SeqHMMLabel(k) = AssignLabel(HMMProbs);  % Assigning initial labels to sequences
end

SeqPerHMM = zeros(1,NHmm);

for k=1:NHmm
        SeqPerHMM(k)=sum(SeqHMMLabel==k); % Number of sequences that get assigned the same label
end

SeqOrigLabel = SeqOrigLabel+1;  % Original labels start from 0
SeqOrigLabel_pruned = SeqOrigLabel;

[ActualCount,~] = hist(SeqOrigLabel,unique(SeqOrigLabel));

M = Mapper(SeqHMMLabel,SeqOrigLabel,NHmm);
Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM' ActualCount'];
   
disp('S.no. Assigned Cluster id - Seq Count Calc - Actual Seq Count');
disp(Table);
   
%% Iterations

HMMRemoved = cell(1);  % HMMs removed due to no sequence assignment
count = 0;
NumItr = 300;

for it = 1:NumItr
    tic
    sprintf('Iteration no = %d\n',it)
    sprintf('No. of HMMs this iteration = %d\n',NHmm)
    %% Assign sequences to HMMs
    
    parfor i = 1:size(X,1)
       Gamma = zeros(1,NHmm);
       Probs = HMMLikelihoodCalC(i-1,NHmm,Pi,X,Aest,mu,cov,Pi_Gmm);
       Probs = exp(Probs - logaddsum(Probs',NHmm));
       
       for hmmInd = 1:NHmm
          Gamma(hmmInd) = (((SeqPerHMM(SeqHMMLabel(i))-1)+ (beta_hmm/NHmm))/(size(X,1)-1+beta_hmm))*(Probs(hmmInd));
       end
       
       SeqHMMLabel(i) = AssignLabel(Gamma/sum(Gamma));
       
     end
         
   
     for k=1:NHmm
         SeqPerHMM(k)=sum(SeqHMMLabel==k); % Number of sequences that get assigned the same label
     end
        
     ZeroLoc =  find(SeqPerHMM==0);      % Find HMMs with no sequences assigned 
        
     if(sum(ZeroLoc) ~= 0)  
         v1=(SeqPerHMM' == 0);
         v2=cumsum(v1);
         SeqPerHMM(v1')=[];
         mu(v1,:) = [];
         cov(v1,:) = [];
         Aest(v1,:) = [];
         Pi_Gmm(v1,:) = [];
        
         SeqHMMLabel=SeqHMMLabel-v2(SeqHMMLabel);           
         NHmm = NHmm - size(ZeroLoc,2);
         count=count+1;
         HMMRemoved{count,:} = ZeroLoc;
         disp('HMMs Removed = ');disp(ZeroLoc);             
     end
        
    M = Cluster_Purity(SeqHMMLabel,SeqOrigLabel_pruned,NHmm,it);
    Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM'];

    disp('S.No. Assigned Cluster id - Sequence Count Calculated');
    disp(Table);   

%% Sample parameters of individual HMMs

    parfor k=1:NHmm    
    
        disp(k);

        SameLabelSeq = X(SeqHMMLabel == k);
        seqlen_new = cellfun(@(x) size(x,2), SameLabelSeq);
     
        if it<151 
            [c1,c2,c3,c4] = BlockSampler(SameLabelSeq,seqlen_new,Aest{k,:},mu{k,:},cov{k,:},beta_Aest(:,:,k),Pi,Pi_Gmm{k,:},globmu,globcov); % Gibbs sampling
        else
            [c1,c2,c3,c4] = BWSampler(SameLabelSeq,Aest{k,:},mu{k,:},cov{k,:},Pi,Pi_Gmm{k,:},k); % Baum Welch sampling
        end
        
        Aest{k,:} = c1;
        mu{k,:} = c2;
        cov{k,:} = c3;
        Pi_Gmm{k,:} = c4;
                
    end
toc
end

end