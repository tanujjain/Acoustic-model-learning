
function [Aest,mu,cov,Pi_Gmm] = BlockSampler(X,seqlen,Aest,mu,cov,beta_Aest,Pi,Pi_Gmm,globmu,globcov)
% Function to implement Block Gibbs sampling to update parameters of an HMM
% where:
% X: All sequences  assigned to this HMM. (num_sequence*1) cell
% seqlen: length of each sequence. (num_sequence*1) cell
% Aest: initial transition matrices. (Nstates*Nstates) matrix for an Hmm with Nstates number of states
% mu: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
% cov: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
% Pi_Gmm: (Nmix*Nstates) matrix containing means for this HMM
% Pi: Initial state probability (same for all HMMs due to left-right assumption)
% globmu, globcov: global mean, covariance. (ObsDim*1) vectors
% return: updated attributes

    Nstates = size(Aest,1);
    allsamplestate=cell(1);
    Num_sequences = size(X,1);
    betadir1_init = beta_Aest(1,:);     % Dirichlet hyper-parameters to sample transition matrix entries
    betadir2_init = beta_Aest(2,:);
    Nmix = size(Pi_Gmm,1);
    
    SamplePerState=zeros(Num_sequences,Nstates);
        
    for n=1:Num_sequences
                                     
       samplestate = BlockStateAssign(X{n,:},seqlen(n),Aest,mu,cov,Nstates,Pi,Pi_Gmm);
                                
        for k=1:Nstates   % Find the number of samples per state
            SamplePerState(n,k)=sum(samplestate == k);
        end
                 
        allsamplestate{n,:} = samplestate;  % Update sample state
    end
    
% Collect all samples belonging to same class in ClassSamples cell 
        
    ClassSamples=cell(Nstates,1);
    
    for st=1:Nstates 
        
             allsamplestateN = cell2mat(allsamplestate');
             XN = cell2mat(X');
             ClassSamples{st} = XN(:,allsamplestateN == st);                                        
    end
    
        
%% Update each Class attributes
    for c=1:Nstates        
        [mu(:,:,c),cov(:,:,c),Pi_Gmm(:,c)] = Sample_GMM(ClassSamples{c},Nmix,mu(:,:,c),cov(:,:,c),Pi_Gmm(:,c),globmu,globcov);
    end    
    
 %% Update Aest
    SameStateTrans =zeros(1,Nstates);         
    InterStateTrans=zeros(1,Nstates-1); % For 3 states: InterStateTrans(1)= state-1 to state-2, InterStateTrans(2)= state-2 to state-3
    for n=1:Num_sequences 
        SameStateTrans = SameStateTrans+SamplePerState(n,:)-ones(1,Nstates); % Same state transitions= number of samples in a class-1
         
        u=1:Nstates-1;
        for j = 1:Nstates-1
            trans=double((allsamplestate{n,:} == u(j))); % trans entries= 1s and 0s
            
            if (find(conv([1 -1],trans)== -1)) % Spot 1 to 0 transitions to see if change has happened
            InterStateTrans(j)=InterStateTrans(j)+1;
            end
            
        end                  
    end         
        
    betadir1 = [SameStateTrans(1) InterStateTrans(1)] + betadir1_init; % Update dirichlet hyperparameters for sampling transition matrix entries
    betadir2 = [SameStateTrans(2) InterStateTrans(2)] + betadir2_init;
    Aest=[dirichlet_gen(betadir1) 0;0 dirichlet_gen(betadir2);0 0 1];          
                     
end


