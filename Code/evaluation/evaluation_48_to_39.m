% Script to get cluster purity for a specific scenario
% Uncomment the part of the code required for supervised/unsupervised,
% segmented/unsegmented cases

clear;

addpath([pwd '\..\Database']);

%% add the folder path where cluster purity should be calculated
%  <Add path here>

NumFiles = 172; %% add total number of files in the folder for evaluation
load X_48phoneme.mat;

ClusterPurity = zeros(1,NumFiles);
%nseqs = zeros(1,NumFiles); % Uncomment if evolution of number of
    %sequences per iteration needs to be seen for the unsegmented case

%% Read in cluster files one by one
instr = 'cluster_it_';
% name = 'cluster_it_94'; % if cluster purity for a single file needs to be
% seen
for i =1:NumFiles
    name = [instr num2str(i)];
    
    %% Unsupervised
    red_mat_u = foldinto39(name);    
    ClusterPurity(i) = sum(max(red_mat_u,[],1))/sum(red_mat_u(:));
    %nseqs(i) = sum(red_mat_u(:)); % Uncomment if evolution of number of
    %sequences per iteration needs to be seen for the unsegmented case
    
    %% Supervised
%     [red_mat,divisor] = foldinto39_supervised(name); 
%     ClusterPurity(i) = sum(max(red_mat,[],1))/divisor; 
end