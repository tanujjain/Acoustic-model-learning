
function [] = ClassifyHMM(X,Aest,mu,cov,NHmm,Pi,SeqOrigLabel,Pi_Gmm,token,it)
% Function to classify validation/training sequences
% where:
%   X: all sequences (validation or training).
%   Aest,mu,cov,Pi_Gmm: Learnt transition matrices, mean, covariance, mixture weights
%   SeqOrigLabel: original labels of the sequences
%   token: 0 if training sequences are classified, 1 if validation sequences are being classified
%   it: iteration number

beta_hmm = 1.0; % Dirichlet parameter for choosing an HMM for a sequence, same for all HMMs
   
HMMProbs = (1/NHmm)*ones(1,NHmm); % InitiaSeqHMMLabell probability kept uniform
SeqHMMLabel = zeros(size(X,1),1);

for k=1:size(X,1)
    SeqHMMLabel(k) = AssignLabel(HMMProbs);  % Assigning initial labels to sequences
end

SeqPerHMM = zeros(1,NHmm);

for k=1:NHmm
        SeqPerHMM(k)=sum(SeqHMMLabel==k); % Number of sequences that get assigned the same label
end

%% Assign labels to each sequence

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
        SeqPerHMM(k)=sum(SeqHMMLabel==k); % Number of sequences that get assigned the same label, to include for display on console
   end
  
   %% Evaluate cluster purity
   
   if token == 0
       disp('Training Results :');
        M = Cluster_Purity_Train(SeqHMMLabel,SeqOrigLabel,NHmm,it);
   else
       disp('Validation Results :');
        M = Cluster_Purity_Valid(SeqHMMLabel,SeqOrigLabel,NHmm,it);
   end
   
   
   Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM'];
   
   disp('S.No. Assigned Cluster id - Sequence Count Calculated - Actual Sequence Count'); % display on console
   disp(Table);
   
end