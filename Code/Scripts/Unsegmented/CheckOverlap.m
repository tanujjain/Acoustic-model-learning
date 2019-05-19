function [label] = CheckOverlap(vec)
% Function to determine the original label of the newly segmented phoneme seqeunce
% where:
%   Vec: vector containing the origial sequence labels for truly segmented sequences. (1*length_of_phoneme_seq)
%   Label: true label of the newly segmented sequence

    [a1,b1] = hist(vec,unique(vec)); % Create histogram of the labels of the samples from truly segmented sequence
    [~,ii] = max(a1); % Get max value in this histogram, which represents the majority original lable samples
    label = b1(ii); % Set label to the max label
end