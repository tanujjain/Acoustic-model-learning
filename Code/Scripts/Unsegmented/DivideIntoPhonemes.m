function [X_ret,Labels] = DivideIntoPhonemes(X_unseg,boundvar,SeqOrigLabel)
% Function to get sequences from the unsegmented TIMIT recording using the boundary variables
%   where:
%   X_Unseg: Unsegmented TIMIT samples for one recording. Matrix (ObsDim*no_of_samples_in_recording)
%   boundvar: boundary variables for this recording. Vector (1*no_of_samples_in_recording)
%   SeqOrigLabel: Original label of the samples in this TIMIT recording. Vector (1*no_of_samples_in_recording)
%   X_ret: cell containing sequences of phonemes determined from this function.
%   Labels: vector containing the original lables of the phonemes with new segmented boundaries. (1*no_of_boundaries)

    X_ret = cell(sum(boundvar,2),1);
    Labels = ones(sum(boundvar,2),1);
    ind = [0 find(boundvar == 1)];
    
    for k = 1:size(ind,2)-1
        X_ret{k,:} = X_unseg(:,ind(k)+1:ind(k+1));
        Labels(k,1) = CheckOverlap(SeqOrigLabel(ind(k)+1:ind(k+1))); % Check maximum overlap of the sequences with original sequences
    end
    
end