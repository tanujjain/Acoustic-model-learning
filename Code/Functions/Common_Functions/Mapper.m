
function [Map] = Mapper(SeqHMMLabel,SeqOrigLabel,NHmm)
% Function to map an HMM to a phoneme label
% by getting the maximum number of true labels 
% for each HMM
% Parameters usage:
% SeqHMMLabel: cluster labels of all sequences 
% SeqOrigLabe: original labels of all sequences
% NHmm: total number of HMMs
% Map: True identity of each HMM

    SeqOrigLabel = SeqOrigLabel+1;  % Original database has classes starting from zero
    Map = zeros(1,NHmm);
    
    for k = 1:NHmm
        assignment = SeqOrigLabel(SeqHMMLabel == k); % Get true labels of sequences assigned to HMM k
        [a,b] = hist(assignment,unique(assignment)); % Count the occurrences of each true class label
        [~,I] = max(a); % get index of the true class with maximum occurrences 
        Map(k) = b(I);  % get the max true class
    end

end