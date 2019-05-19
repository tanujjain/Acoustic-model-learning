
function [Aest,mu,cov,Pi_Gmm,SeqPerHMM] = Multi_HMM_Sampler_Unseg_cont_it(X_Unseg,AbsAudioBoundaryInd,Nmix,globmu,globcov,Nstates,SeqOrigLabel,SeqPerHMM_old_orig,...
    SeqHMMLabel_old,Seq_per_rec_inds_old,Aest,mu,cov,Pi_Gmm,Total_Pi,Total_trans_mat,iteration,NumIter,NHmm)
% Function to implement sampling of attributes for unsegmented recordings with segmentation algorithm
% where:
%   X_Unseg: All samples in all recordings. Matrix (ObsDim*no_of_samples_in_all_recordings)
%   Nmix: number of mixtures
%   Nstates: number of states
%   SeqOrigLabel: Original class labels of the sequences. (Num_sequences*1) vector
%   globmu, globcov: global mean, covariance. (ObsDim*1) vectors
%   AbsAudioBoundaryInd: Contains the sample numbers of the last sample in a TIMIT recording for all TIMIT 	recordings,vector of size (1*4621)
%   SeqPerHMM_old_orig: SeqPerHMM from the last iteration
%   SeqHMMLabel_old: Sequence labels from last iteration
%   Seq_per_rec_inds_old: cumulative sum of the number of sequences discovered per recording in last iteration. Vector (1*(sequences_per_recording+1)), first element as zero
%   iteration: iteration from which the execution begins
%   NumIter: Total number of iterations
%   NHmm: Total number of HMMs discovered in the last iteration
%   return:
%       Aest: Transition matrices. (NHmm*1) cell with (Nstates*Nstates) matrix for Hmms with Nstates number of states
%       mu: NHmm by 1 cell, contains means, each cell contains (ObsDim*Nmix*Nstates) matrix
%       cov: NHmm by 1 cell, contains covariances, each cell contains (ObsDim*Nmix*Nstates) matrix
%       Pi_Gmm: NHmm by 1 cell, contains mixture weights, each cell contains (Nmix*Nstates) matrix
%       SeqPerHMM: number of sequences per HMM. (NHmm*1) vector

%% Get Attributes

ObsDim=size(X_Unseg,1);
Pi = [1 0 0];

%% HMM Variables
SeqPerHMM = SeqPerHMM_old_orig;
Nseqs_old = sum(SeqPerHMM);
NHmm_old = NHmm;  % number of HMMs in last iteration

%% Initialize parameters
beta_hmm = 1;                     % Dirichlet Process parameter
beta_Aest = ones(Nstates-1,Nstates-1);

%% Begin Iterations

for it=iteration:NumIter
 tic  
    disp(it);
    %% Get boundaries
    
    BoundVars = Get_Segment_boundaries(X_Unseg,AbsAudioBoundaryInd,NHmm,mu,cov,Pi_Gmm,Total_Pi,Total_trans_mat);
    save('boundvars.mat','BoundVars');

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
   disp('Sequences Obtained');
   
   SeqHMMLabel = zeros(Nseqs,1);
   SeqPerHMM = zeros(1,NHmm);
   
    %%  Clustering

    for i=1:Nseqs                       
        rec_no = find(i <= Seq_per_rec_inds,1,'first')-1;  % Find record number which contains this sequence
            
        save('seq_rec_ind.mat','Seq_per_rec_inds','SeqHMMLabel_old','i');
        inds_to_reduce = SeqHMMLabel_old(Seq_per_rec_inds_old(rec_no)+1:Seq_per_rec_inds_old(rec_no+1)); % remove the samples belonging to this label
                                                                                                         % recording for evaluating prior terms for sampling    
        u_red = unique(inds_to_reduce);  % HMM labels to be reduced
                
        total_red_count = 0;  % number of samples removed
                
        if NHmm_old < NHmm % if no of HMMs exceed the count in last iteration, take into account the newly discovered HMMs also 
            SeqPerHMM_old = [SeqPerHMM_old_orig SeqPerHMM(NHmm_old+1:end)];
            total_red_count = -sum(SeqPerHMM_old(NHmm_old+1:end));
        else
            SeqPerHMM_old = SeqPerHMM_old_orig;
        end
                
        for ii = 1:size(u_red,1)
            red_count = sum(u_red(ii) == inds_to_reduce);
            SeqPerHMM_old(u_red(ii)) = SeqPerHMM_old(u_red(ii))-red_count;
            if SeqPerHMM_old(u_red(ii)) == 0
                SeqPerHMM_old(u_red(ii)) = 1;
                red_count = red_count+1;
            end
            total_red_count = total_red_count+red_count;
        end
        Nseqs_updated = Nseqs_old-total_red_count;  % Update number of sequences
        
 % Calculate posterior
 
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
            Gamma(c)=(SeqPerHMM_old(c)/(Nseqs_updated-1+beta_hmm))*Probs(c);
        end
        
        Gamma(end) = (beta_hmm/(Nseqs_updated-1+beta_hmm))*Probs(end);
        
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
    disp('Updating parameters ...');

    Xsliced = cell(NHmm,1);
    State_three_Prob = zeros(NHmm,1);

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
    
%% Build total transition matrix  and Total_Pi matrix
    Total_trans_mat = blkdiag(Aest{:});
    ind_to_mod =(1:3:NHmm*3);
    
    for k = 1:NHmm
        Prob_HMM = (1-State_three_Prob(k)/10)*(SeqPerHMM/sum(SeqPerHMM,2));
        mod_vec = zeros(1,NHmm*3);
        mod_vec(ind_to_mod) = Prob_HMM;
        Total_trans_mat(k*3,:) = mod_vec;
        Total_trans_mat(3*k,3*k) = State_three_Prob(k)/10;
    end
    
    Total_Pi = kron(SeqPerHMM'/sum(SeqPerHMM,2),Pi');
    clear Xsliced;   

    %% Update variables for the next iteration        
    SeqHMMLabel_old = SeqHMMLabel;
    Nseqs_old = Nseqs;
    SeqPerHMM_old_orig = SeqPerHMM;
    Seq_per_rec_inds_old = Seq_per_rec_inds;
    NHmm_old = NHmm;
        
    disp('Num of HMMs:');
    disp(NHmm);
            
    name = ['it_' num2str(it) '.mat'];
    save(name,'Aest','mu','cov','Pi_Gmm','SeqHMMLabel','SeqOrigLabel_per_Seq','SeqPerHMM','NHmm','it','State_three_Prob',...
            'Seq_per_rec_inds','rec_no','Total_Pi','Total_trans_mat','SeqPerHMM_old');
    
    if it>2
        NameToDelete = ['it_' num2str(it-2) '.mat'];
        delete(NameToDelete);
    end           
        toc
    clear X;

end

end