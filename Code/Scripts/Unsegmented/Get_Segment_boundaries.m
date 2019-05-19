
function [BoundVars] = Get_Segment_boundaries(X_Unseg,AbsAudioBoundaryIndApp,NHmm,mu,cov,Pi_Gmm,Total_Pi,Total_trans_mat)
% Function to implement segmentation algorithm
% where,
%   X_Unseg: Matrix containing all samples from all recordings. (ObsDim * total_no_of_samples_in_all_recordings)
%   AbsAudioBoundaryIndApp: Indices of the last sample in a TIMIT recording. (1*4621) vector, explained in readme document
%   NHmm: number of HMMs
%   Aest: initial transition matrices. (Nstates*Nstates) matrix for an Hmm with Nstates number of states
%   mu: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
%   cov: (ObsDim*Nmix*Nstates) matrix containing means for this HMM
%   Pi_Gmm: (Nmix*Nstates) matrix containing means for this HMM
%   Total_Pi: Initial state probability of being in different states of cascaded HMMs. vector with (NHmm*Nstates) entries
%   Total_trans_mat: Total (cascaded) transition matrix of the complete system. Matrix(NHmm*Nstates rows, NHmm*Nstates columns)
%   BoundVars: Boundary variables for each recording, (4620*1) cell


%reshape mu,cov,Pi_Gmm to pass to function Segment_Per_Audio
    meanmat = zeros(size(X_Unseg,1),size(Pi_Gmm{1},1),NHmm*size(Pi_Gmm{1},2));
    covmat = zeros(size(X_Unseg,1),size(Pi_Gmm{1},1),NHmm*size(Pi_Gmm{1},2));
    Pigmmmat = zeros(size(Pi_Gmm{1},1),NHmm*size(Pi_Gmm{1},2)); 


    for k=1:NHmm
        meanmat(:,:,3*k-2:3*k)=mu{k};
        covmat(:,:,3*k-2:3*k)=cov{k};
        Pigmmmat(:,3*k-2:3*k) = Pi_Gmm{k};
    end
    
    BoundVars = cell(4620,1);
    
    parfor k = 1:size(AbsAudioBoundaryIndApp,2)-1

        start = AbsAudioBoundaryIndApp(k)+1; % To get start and end indices for each recording
        last =  AbsAudioBoundaryIndApp(k+1);

        boundary = Segment_Per_Audio(X_Unseg(:,start:last),meanmat,covmat,Pigmmmat,Total_Pi,Total_trans_mat); % Call segmentation function
        BoundVars(k,:) = mat2cell(boundary);
    end
    
    poolobj = gcp('nocreate'); % Close parallel pool
    delete(poolobj);
    
end