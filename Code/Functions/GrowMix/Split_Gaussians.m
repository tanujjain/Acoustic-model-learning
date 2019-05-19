function [muout,covout,Pi_Gmmout] = Split_Gaussians(X,seqlen,mu,cov,Pi_Gmm,Aest)
% Function to split gaussian components of the GMM of each state
% where:
%   X: All sequences belonging to an HMM. (number_of_sequences*1) cell with each cell (ObsDim*sequence_length) matrix
%   seqlen: length of each sequence in X.(number_of_sequences*1) vector
%   mu: means of each GMM compenent of each state. (ObsDim*Nmix*Nstates) matrix
%   cov: covariances of each GMM compenent of each state. (ObsDim*Nmix*Nstates) matrix
%   Pi_Gmm: mixture weights of each GMM component. (Nmix*Nstates) matrix
%   Aest: Transition probabilities for this HMM. (Nstates*Nstates) matrix

Nstates = size(mu,3);
Pi = [1 0 0];

% get state of eah sample in each sequence
allsamplestate = cellfun(@(x,y) BlockStateAssign(x,y,Aest,mu,cov,Nstates,Pi,Pi_Gmm),X,num2cell(seqlen),'UniformOutput',0);
         
%% Collect all samples belonging to same class in ClassSamples cell 
        
    ClassSamples=cell(Nstates,1);
    
    for st=1:Nstates        
        allsamplestateN = cell2mat(allsamplestate');
        XN = cell2mat(X');
        ClassSamples{st} = XN(:,allsamplestateN == st);                                        
    end
      
    muout = zeros(size(mu,1),size(mu,2)*2,size(mu,3));
    covout = zeros(size(mu,1),size(mu,2)*2,size(mu,3));
    Pi_Gmmout = zeros(size(mu,2)*2,size(mu,3));
  
 %% Use samples for each state to cary out splitting process for that state   
    for st = 1:3
        [out1,out2,out3] = Split_State(ClassSamples{st,:},mu(:,:,st),cov(:,:,st),Pi_Gmm(:,st));
         muout(:,:,st) = out1;
         covout(:,:,st) = out2;
         Pi_Gmmout(:,st) = out3;
    end    
end