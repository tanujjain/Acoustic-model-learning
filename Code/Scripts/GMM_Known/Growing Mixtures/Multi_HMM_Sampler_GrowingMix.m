
function [Aest,mu,cov,SeqHMMLabel,HMMRemoved] = Multi_HMM_Sampler_GrowingMix(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm,globmu,globcov)
% Function to implement Gibbs sampling
% where:
%   X: All sequences
%   Aest, mu, cov,Pi_Gmm: transition matrix, mean, covariance, mixture weights for GMMs
%   NHmm: number of HMMs
%   Pi: Initial state probability (same for all HMMs due to left-right assumption)
%   SeqOrigLabel: Original class labels of the sequences
%   globmu, globcov: global mean, covariance
%   return: updated attributes

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

SeqOrigLabel = SeqOrigLabel+1;
SeqOrigLabel_pruned = SeqOrigLabel;

[ActualCount,~] = hist(SeqOrigLabel,unique(SeqOrigLabel));

M = Mapper(SeqHMMLabel,SeqOrigLabel,NHmm);
Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM' ActualCount'];
   
disp('S.no. Assigned Cluster id - Seq Count Calc - Actual Seq Count');
disp(Table);
   
   %% Iterations
% HMMRemoved = zeros(1,1);
HMMRemoved = cell(1);
count = 0;
NumItr = 400;

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
            try
            SeqHMMLabel(i) = AssignLabel(Gamma/sum(Gamma));
            catch
                disp('Issue in MultiHMM');
                disp('Gamma = '); disp(Gamma);
                disp('SeqHMMLabel(i) = '); disp(SeqHMMLabel(i));
                disp('Probs = '); disp(Probs);
                disp('Seq no. = '); disp(i);                              
            end
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
        
   M = Cluster_Purity_growing(SeqHMMLabel,SeqOrigLabel_pruned,NHmm,it); % Evaluate cluster purity
   Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM'];

   disp('S.No. Assigned Cluster id - Sequence Count Calculated');
   disp(Table);   

%% Sample parameters of individual HMMs

   parfor k=1:NHmm    
    
        disp(k);

        SameLabelSeq = X(SeqHMMLabel == k);
        seqlen_new = cellfun(@(x) size(x,2), SameLabelSeq);
               

        [c1,c2,c3,c4] = BlockSampler(SameLabelSeq,seqlen_new,Aest{k,:},mu{k,:},cov{k,:},beta_Aest(:,:,k),Pi,Pi_Gmm{k,:},globmu,globcov);

         Aest{k,:} = c1;
         mu{k,:} = c2;
         cov{k,:} = c3;
         Pi_Gmm{k,:} = c4;
                
   end
    
   if it == 51 || it == 101 || it == 151 || it == 201 || it == 251  % Split the Gaussians every 50 iterations
       parfor k= 1:NHmm
           disp(k);    
           SameLabelSeq = X(SeqHMMLabel == k);
           seqlen_new = cellfun(@(x) size(x,2), SameLabelSeq);
           [c1,c2,c3] = Split_Gaussians(SameLabelSeq,seqlen_new,mu{k,:},cov{k,:},Pi_Gmm{k,:},Aest{k,:});
           mu{k,:} = c1;
           cov{k,:} = c2;
           Pi_Gmm{k,:} = c3;
       end
   end
   
toc
end

end