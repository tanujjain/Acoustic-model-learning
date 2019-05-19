
function [Aest,mu,cov,Pi_Gmm,SeqHMMLabel,HMMRemoved] = Baum_Welch_GrowMix(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm)
% Function to implement increasing mixture comüonents using Baum Welch algorithm 
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
%   HMMRemoved: keeps track of HMMs removed (if any) with each iteration. cell structure (never got any values for any simulation, was just here to check)
%   return: updated attributes + cell containing a list indices of removed HMMs + SeqHMMLabel


%% Hyper Parameter Initialization

beta_hmm = 1.0; % Dirichlet parameter for choosing an HMM for a sequence, same for all HMMs

HMMProbs = (1/NHmm)*ones(1,NHmm); % Initial SeqHMMLabel probability kept uniform
SeqHMMLabel = zeros(size(X,1),1);

for k=1:size(X,1)
    SeqHMMLabel(k) = AssignLabel(HMMProbs);  % Assigning initial labels to sequences
end

SeqPerHMM = zeros(1,NHmm);

for k=1:NHmm
        SeqPerHMM(k)=sum(SeqHMMLabel==k); % Number of sequences that get assigned the same label
end

SeqOrigLabel = SeqOrigLabel+1;  % Original sequence label start from 0
SeqOrigLabel_pruned = SeqOrigLabel;

[ActualCount,~] = hist(SeqOrigLabel,unique(SeqOrigLabel));

M = Mapper(SeqHMMLabel,SeqOrigLabel,NHmm);
Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM' ActualCount'];
   
disp('S.no. Assigned Cluster id - Seq Count Calc - Actual Seq Count');
disp(Table);
   
   %% Iterations

HMMRemoved = cell(1);
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
            try
            SeqHMMLabel(i) = AssignLabel(Gamma/sum(Gamma));
            catch
                disp('Issue in MultiHMM');
                disp('Gamma = '); disp(Gamma);
                disp('SeqHMMLabel(i) = '); disp(SeqHMMLabel(i));
                disp('Probs = '); disp(Probs);
                disp('Seq no. = '); disp(i);      
                disp('Iteration no. = '); disp(it);
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
        
   M = Cluster_Purity(SeqHMMLabel,SeqOrigLabel_pruned,NHmm,it);
   Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM'];

   disp('S.No. Assigned Cluster id - Sequence Count Calculated');
   disp(Table);   

%% Sample parameters of individual HMMs

   parfor k=1:NHmm    
   
        disp(k);

        [c1,c2,c3,c4] = BWSampler(X(SeqHMMLabel == k),Aest{k,:},mu{k,:},cov{k,:},Pi,Pi_Gmm{k,:},k);
        Aest{k,:} = c1;
        mu{k,:} = c2;
        cov{k,:} = c3;
        Pi_Gmm{k,:} = c4;
                        
   end
    
     if it == 51 || it == 101 || it == 151 || it == 201 || it == 251
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