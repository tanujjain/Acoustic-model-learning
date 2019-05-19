function [label]=AssignLabel(gamma)
% Function to assign label given a set of posterior probabilities; performs a sampling 
% from a categorical distribution
%
% where:
%   gamma: posterior probabilites vector
%   label: assigned label

a=cumsum(gamma);
y=rand; 

label=find(y<=a,1,'first'); % find label by getting the first occurrence of a value greater than y
end