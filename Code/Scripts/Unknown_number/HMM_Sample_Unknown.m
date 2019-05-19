
function [Aest,mu,cov,Pi_Gmm,SeqPerHMM] = HMM_Sample_Unknown(X,Nmix,globmu,globcov,Nstates,SeqOrigLabel)
% Function to implement sampling of attributes for unknown number of HMM case 
% where:
%   X: All sequences. (Num_sequences*1) cell
%   Nmix: number of mixtures
%   Nstates: number of states
%   SeqOrigLabel: Original class labels of the sequences. (Num_sequences*1) vector
%   globmu, globcov: global mean, covariance. (ObsDim*1) vectors
%   return:
%       Aest: Transition matrices. (NHmm*1) cell with (Nstates*Nstates) matrix for Hmms with Nstates number of states
%       mu: NHmm by 1 cell, contains means, each cell contains (ObsDim*Nmix*Nstates) matrix
%       cov: NHmm by 1 cell, contains covariances, each cell contains (ObsDim*Nmix*Nstates) matrix
%       Pi_Gmm: NHmm by 1 cell, contains mixture weights, each cell contains (Nmix*Nstates) matrix
%       SeqPerHMM: number of sequences per HMM. (NHmm*1) vector


%% Get Attributes
Nseqs = size(X,1);
ObsDim=size(X{1,:},1);
Pi = [1 0 0];
SeqOrigLabel = SeqOrigLabel+1;

%% HMM Variables
NHmm=0;
SeqPerHMM = zeros(1,NHmm);
SeqHMMLabel = zeros(Nseqs,1);

%% Initialize parameters
beta_hmm = 1;   % Dirichlet Process hyperparameter

Aest = cell(1);
mu = cell(1);
cov = cell(1);
Pi_Gmm = cell(1);

%% Begin Iterations
NumIter = 200;
for it=1:NumIter
  tic  
    disp(it);
    for i=1:Nseqs
        if(SeqHMMLabel(i)~=0)
            SeqPerHMM(SeqHMMLabel(i)) = SeqPerHMM(SeqHMMLabel(i))-1;
            SeqHMMLabel(i)=0;
        end
        
 %% Calculate posterior
 disp(i)
        Gamma=zeros(1,NHmm+1);
    
   % Sample parameters for new mixture
        AestNew = zeros(Nstates,Nstates);   
        MeanNew = zeros(ObsDim,Nmix,Nstates);
        CovNew = zeros(ObsDim,Nmix,Nstates);
        Pi_GmmNew = zeros(Nmix,Nstates);

        for st = 1:Nstates
            [MeanNew(:,:,st),CovNew(:,:,st),Pi_GmmNew(:,st),AestNew] = ...
                            Sample_GMM_Prior(globmu,globcov,Nmix,Nstates);
        end

        mu{end+1,:} = MeanNew; % Add newly sampled attributes to the end of cell containing all attributes
        cov{end+1,:} = CovNew;
        Pi_Gmm{end+1,:} = Pi_GmmNew;
        Aest{end+1,:} = AestNew;
    
        if i == 1 && it == 1
            mu(1,:) = [];
            cov(1,:) = [];
            Pi_Gmm(1,:) = [];
            Aest(1,:) = [];
        end
 
        Probs = hmm_like_mthread_stack(i-1,NHmm+1,Pi,X,Aest,mu,cov,Pi_Gmm);
    
        Probs = exp(Probs - logaddsum(Probs',NHmm+1));
      
        for c = 1:NHmm
            Gamma(c)=(SeqPerHMM(c)/(Nseqs-1+beta_hmm))*Probs(c);
        end
        
        Gamma(end) = (beta_hmm/(Nseqs-1+beta_hmm))*Probs(end);
        
 % Attach label to the sequence based on the evaluated posterior
 
        SeqHMMLabel(i) = AssignLabel(Gamma/sum(Gamma));
        
        if SeqHMMLabel(i)==(NHmm+1)
            NHmm=NHmm+1;
            SeqPerHMM(NHmm)=1;
        else
            SeqPerHMM(SeqHMMLabel(i))= SeqPerHMM(SeqHMMLabel(i))+1;
            mu(end,:) = [];
            cov(end,:) = [];
            Aest(end,:) = [];
            Pi_Gmm(end,:) = [];
            clear MeanNew;
            clear CovNew;
            clear Pi_GmmNew;
            clear AestNew;
        end
     
        clear Probs;
        clear Gamma;
    end
    
    %% Remove additional Unassigned labels
          
    u=unique(SeqHMMLabel);
    NHmm=size(u,1);
        
    v1=(SeqPerHMM' == 0);
    v2=cumsum(v1);
    SeqPerHMM(v1')=[];
    mu(v1,:) = [];
    cov(v1,:) = [];
    Aest(v1,:) = [];
    Pi_Gmm(v1,:) = [];
        
    SeqHMMLabel=SeqHMMLabel-v2(SeqHMMLabel);
    
%% Cluster Purity

   M = Cluster_Purity_Unknown(SeqHMMLabel,SeqOrigLabel,NHmm,it);
   Table = [linspace(1,NHmm,NHmm)' M' SeqPerHMM'];

   disp('S.No. Assigned Cluster id - Sequence Count Calculated');
   disp(Table);
   
%% Update parameters for next iteration
    Xsliced = cell(NHmm,1);

    for k=1:NHmm
        Xsliced{k,:} = X(SeqHMMLabel == k); 
    end
    poolobj = parpool;
    
    parfor k =1:NHmm       
        disp(k);

        seqlen_new = cellfun(@(x) size(x,2), Xsliced{k,:});
        [c1,c2,c3,c4] = BlockSampler_Unknown(Xsliced{k,:},seqlen_new,Aest{k,:},mu{k,:},cov{k,:},beta_Aest,Pi,Pi_Gmm{k,:},globmu,globcov);
                
        Aest{k,:} = c1;
        mu{k,:} = c2;
        cov{k,:} = c3;
        Pi_Gmm{k,:} = c4;     
    end
    
    % delete parallel pool cluster and clear unneeded variables
    delete(poolobj); 
    clear Xsliced;    
    disp('Num of HMMs:');
    disp(NHmm);
    
    name = ['it_' num2str(it) '.mat']; % Retain attributes of only the latest two iterations
    save(name,'Aest','mu','cov','Pi_Gmm','SeqHMMLabel','SeqPerHMM','NHmm','it');
    
    if it>2
    NameToDelete = ['it_' num2str(it-2) '.mat'];
    delete(NameToDelete);
    end
    
    toc
end

end