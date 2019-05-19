function [Aest,mu,cov,globmu,globcov,Pi_Gmm,SeqHMMLabel] = flat_HMM_rand(Xseq,NHmm,Nstates,Nmix_gmm)
% Function for flat start initialization of HMM attributes for all HMMs
% where:
%       Xseq:  All sequences. (num_sequences*1) cell with each cell containing (ObsDim*sequence_length) elements
%       NHmm: number of HMMs
%       Nstates: number of states
%       Nmix_gmm: number of mixtures per GMM
% return:
%       Aest: Transition matrices. (NHmm*1) cell with (Nstates*Nstates) matrix for Hmms with Nstates number of states
%       mu: NHmm by 1 cell, contains means, each cell contains (ObsDim*Nmix*Nstates) matrix
%       cov: NHmm by 1 cell, contains covariances, each cell contains (ObsDim*Nmix*Nstates) matrix
%       Pi_Gmm: NHmm by 1 cell, contains mixture weights, each cell contains (Nmix*Nstates) matrix
%       Pi: Initial state probability (same for all HMMs due to left-right assumption)
%       SeqHMMLabel: Assigned cluster labels of the sequences. (Num_sequences*1) vector
%       globmu, globcov: global mean, covariance. (NHmm*1) cell with each cell element having (ObsDim*1) vector


Aest = cell(NHmm,1);
mu = cell(NHmm,1);
cov = cell(NHmm,1);
Pi_Gmm = cell(NHmm,1);
globmu = cell(NHmm,1);
globcov = cell(NHmm,1);

SeqHMMLabel = zeros(size(Xseq,1),1);

HMMProbs = (1/NHmm)*ones(1,NHmm); % Initial SeqHMMLabel probability kept uniform

for k=1:size(Xseq,1)
    SeqHMMLabel(k) = AssignLabel(HMMProbs);  % Assigning initial labels to sequences
end

    for k=1:NHmm    
        SameLabelSeq = Xseq(SeqHMMLabel == k);
        seqlen_init = cellfun(@(x) size(x,2), SameLabelSeq);
        [Aest{k,:},mu{k,:},cov{k,:},Pi_Gmm{k,:},globmu{k,:},globcov{k,:}] = flatstart_init_func_rand(SameLabelSeq,seqlen_init',Nstates,Nmix_gmm);     
    end

end
