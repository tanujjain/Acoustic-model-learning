function [Aest,mu,cov,Pi_Gmm,SeqPerHMM] = Multi_HMM_Sampler_Unsegmented(X_Unseg,AbsAudioBoundaryInd,Nmix,globmu,globcov,Nstates,SeqOrigLabel)
% Function to implement sampling of attributes for unsegmented recordings 
% where:
%   X_Unseg: All samples in all recordings. Matrix (ObsDim*no_of_samples_in_all_recordings)
%   Nmix: number of mixtures
%   Nstates: number of states
%   SeqOrigLabel: Original class labels of the sequences. (Num_sequences*1) vector
%   globmu, globcov: global mean, covariance. (ObsDim*1) vectors
%   AbsAudioBoundaryInd: Contains the sample numbers of the last sample in a TIMIT recording for all TIMIT 	recordings,vector of size (1*4621)  
%   return:
%       Aest: Transition matrices. (NHmm*1) cell with (Nstates*Nstates) matrix for Hmms with Nstates number of states
%       mu: NHmm by 1 cell, contains means, each cell contains (ObsDim*Nmix*Nstates) matrix
%       cov: NHmm by 1 cell, contains covariances, each cell contains (ObsDim*Nmix*Nstates) matrix
%       Pi_Gmm: NHmm by 1 cell, contains mixture weights, each cell contains (Nmix*Nstates) matrix
%       SeqPerHMM: number of sequences per HMM. (NHmm*1) vector


BoundVars = PreSegFunc(X_Unseg,AbsAudioBoundaryInd); %% Use presegmentation algorithm to get boundary variables

    %%  Construct sequences from boundaries
    
    Seq_per_rec = cellfun(@(y) sum(y,2),BoundVars);
    Nseqs = sum(Seq_per_rec);
    
    Seq_per_rec_inds = [0;cumsum(Seq_per_rec)];
    
    X = cell(Nseqs,1);
    SeqOrigLabel_per_Seq = ones(Nseqs,1);
    
   for seqid = 1:size(BoundVars,1)
       start = AbsAudioBoundaryInd(seqid)+1;
       last =  AbsAudioBoundaryInd(seqid+1);
       
       start_ind_X = Seq_per_rec_inds(seqid)+1;
       last_ind_X = Seq_per_rec_inds(seqid+1);
       [D,S] = DivideIntoPhonemes(X_Unseg(:,start:last),BoundVars{seqid,:},SeqOrigLabel{seqid,:});
       X(start_ind_X:last_ind_X) = D;
       SeqOrigLabel_per_Seq(start_ind_X:last_ind_X) = S;
   end
      
   SeqHMMLabel = zeros(Nseqs,1);

%% Get Attributes

ObsDim=size(X{1,:},1);
Pi = [1 0 0];

%% HMM Variables
NHmm=0;
SeqPerHMM = zeros(1,NHmm);


%% Initialize parameters
beta_hmm = 1;                     % Dirichlet Process parameter
beta_Aest = ones(Nstates-1,Nstates-1);

Aest = cell(1);
mu = cell(1);
cov = cell(1);
Pi_Gmm = cell(1);


%% Begin Iterations
NumIter=20;  % Set number of iterations tuning must be done after getting boundaries from presegmentation step
for it=1:NumIter
  tic  
    disp(it);
    for i=1:Nseqs
        if(SeqHMMLabel(i)~=0)
            SeqPerHMM(SeqHMMLabel(i)) = SeqPerHMM(SeqHMMLabel(i))-1;
            SeqHMMLabel(i)=0;
        end
        
 %% Calculate posterior
 
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

        mu{end+1,:} = MeanNew;
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

   M = Cluster_Purity_Unseg(SeqHMMLabel,SeqOrigLabel_per_Seq,NHmm,it);
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
        [c1,c2,c3,c4,c5] = BlockSampler_unseg(Xsliced{k,:},seqlen_new,Aest{k,:},mu{k,:},cov{k,:},beta_Aest,Pi,Pi_Gmm{k,:},globmu,globcov);
        Aest{k,:} = c1;
        mu{k,:} = c2;
        cov{k,:} = c3;
        Pi_Gmm{k,:} = c4;
        State_three_Prob(k) = c5;
    end
    delete(poolobj);
    clear Xsliced;    
    disp('Num of HMMs:');
    disp(NHmm);
    
    name = ['it_' num2str(it) '.mat'];
    save(name,'Aest','mu','cov','Pi_Gmm','SeqHMMLabel','SeqPerHMM','Seq_per_rec_inds','NHmm','it');
    
    if it>2
    NameToDelete = ['it_' num2str(it-2) '.mat'];
    delete(NameToDelete);
    end
    
    toc
end

%% Build tital transition matrix (cascaded matrix) and total_Pi vector

Total_trans_mat = blkdiag(Aest{:});
ind_to_mod =(1:3:NHmm*3);
    
for k = 1:NHmm
    Prob_HMM = (1-State_three_Prob(k)/2)*(SeqPerHMM/sum(SeqPerHMM,2));
    mod_vec = zeros(1,NHmm*3);
    mod_vec(ind_to_mod) = Prob_HMM;
    Total_trans_mat(k*3,:) = mod_vec;
    Total_trans_mat(3*k,3*k) = State_three_Prob(k)/2;
end
    
Total_Pi = kron(SeqPerHMM'/sum(SeqPerHMM,2),Pi');d
        
%% Call function for running for subsequent iterations of the algorithm using segmentation algorithm 

[Aest,mu,cov,Pi_Gmm,SeqPerHMM] = Multi_HMM_Sampler_Unseg_cont_it(X_Unseg,AbsAudioBoundaryInd,Nmix,globmu,globcov,Nstates,SeqOrigLabel,SeqPerHMM,...
    SeqHMMLabel,Seq_per_rec_inds,Aest,mu,cov,Pi_Gmm,Total_Pi,Total_trans_mat,2,1000,NHmm);

end