
function [StateSeq] = BlockStateAssign(X,seqlen,Aest,mu,cov,Nstates,Pi,Pi_Gmm)
% Function to do block Gibbs sampling (or Backward sampling) of the states for a given sequence
% where:
%   X: sequence whose samples need to be assigned a state. (ObsDim*sequence_length) matrix
%   seqlen: length of this sequence.
%   Aest: Transition matrix for the HMM this sequence belongs to. (Nstates*Nstates) matrix
%   mu: means for the HMM this sequence belongs to. (ObsDim*Nmix*Nstates) matrix
%   cov: covariance for the HMM this sequence belongs to. (ObsDim*Nmix*Nstates) matrix
%   Pi: Initial state probabilities vector with value[1 0 0]
%   Pi_Gmm: Mixture weights for all mixture components for all states. (Nmix*Nstates) matrix
%   Nstates: number of states
% return:
%   Stateseq: sequence of states.(1*seqlen) vector

logA=log(Aest);
        
B = State_Obs_Prob_GMM(X,mu,cov,Nstates,Pi_Gmm); % Observation probability for each sample and state combination

logPi=log(Pi)';

        
%% Forward Calculation  

[logAlpha,~] = ForwardAlgorithmLog(B,logA,logPi); % get forward variable for the lattice

 %% Back Sampling
 StateSeq = zeros(1,seqlen);
 StateSeq(end) = Nstates;
 
 for i = seqlen-1:-1:2  % goes over all samples except the last and first in the sequence
       G = logA(:,StateSeq(i+1)) + logAlpha(:,i); % transition_probs*alpha
       G = exp(G - logaddsum(G,Nstates)); % normalization of the vector G
       StateSeq(i) = AssignLabel(G);
 end
 
 StateSeq(1) = 1;
 
end