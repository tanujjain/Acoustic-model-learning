
function [Aest,mu,cov,Pi_Gmm,globmu,globcov] = flatstart_init_func_rand(X,seqlen,Nstates,Nmix_gmm)
% Function for flat start initialization of HMM attributes for one HMM
% where:
%       X:  All sequences corresponding to this HMM. (num_sequences_assigned_to_hmm*1) cell
%       seqlen: length of all sequences assigned to this HMM. (num_sequences_assigned_to_hmm*1) cell
%       Nstates: number of states
%       Nmix_gmm: number of mixtures per GMM
% return: Initialised attributes. globmu, globcov: for each state
%       Aest: initial transition matrices. (Nstates*Nstates) matrix for an Hmm with Nstates number of states
%       mu: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
%       cov: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
%       Pi_Gmm: (Nmix*Nstates) matrix containing means for this HMM
%       globmu: (ObsDim*Nstates) matrix containing global mean for each class of this HMM 
%       globcov: (ObsDim*Nstates) matrix containing global covariance for each class of this HMM 


    Xnew=cell(Nstates,1);  % for all sequences assigned to a state
    mu = zeros(size(X{1,:},1),Nmix_gmm,Nstates);
    cov = zeros(size(X{1,:},1),Nmix_gmm,Nstates);
    Pi_Gmm = zeros(Nmix_gmm,Nstates);
    globmu = zeros(size(X{1,:},1),Nstates);
    globcov = zeros(size(X{1,:},1),Nstates);
    
    SamplesPerState = zeros(1,Nstates);
    SampleStateDeno = zeros(1,Nstates);
    
    for ind =1:size(seqlen,2)
        
        l=floor(seqlen(ind)/Nstates);
        
        o=X{ind,:};
        beg=0;
 % Divide each sequence such that each part contains length/Nstates samples     
        for st = 1:Nstates
            beg=beg+1;
            last = st*l;
             increment = l - 1;
            if st == Nstates
                last= last + mod(seqlen(ind),Nstates);
                increment = l + mod(seqlen(ind),Nstates) - 1;
            end
            Xnew{st,:} = cat(2,Xnew{st,:},o(:,beg:last));
            beg=last;            
           SamplesPerState(st) = SamplesPerState(st) + increment;
           SampleStateDeno(st) = SampleStateDeno(st) + increment + 1;
        end
       
    end
   
      SamplesPerState(Nstates) = SamplesPerState(Nstates) + size(seqlen,2);
%% Get mean, covariance, mixture weights for each state. Also get global means and covariaces for each state     
    for p=1:Nstates
        globmu(:,p)=sum(Xnew{p,:},2)/size(Xnew{p,:},2);
        globcov(:,p)=sum((Xnew{p,:}-repmat(globmu(:,p),1,size(Xnew{p,:},2))).^2,2)/size(Xnew{p,:},2);
        
        [Pi_Gmm(:,p),mu(:,:,p),cov(:,:,p)] = Gmm_attributes_rand(Xnew{p,:},Nmix_gmm);       
    end

%% Get Aest using the number of samples assigned to each state 
    vec1 = SamplesPerState./SampleStateDeno;
    vec2 = repmat(size(seqlen,2),1,Nstates-1)./SampleStateDeno(1,1:end-1);
    
    Aest1 = diag(vec1);
    Aest2 = diag(vec2,1);
    
    Aest = Aest1 + Aest2;
end