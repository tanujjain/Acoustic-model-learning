function [inpFile] = foldinto39(name)
% Function to fold phonemes from 48 to 39 for supervised case(only
% discovered labels to be folded)

%% read cluster_purity file
inpFile = dlmread(name);
inpFile = inpFile(3:end,:);

%% Indices for phoneme

inDtoFold=[3 5 8 14 15 16 24 28 47];
FoldInto = [0 2 21 29 31 21 23 21 38];

%% Incremented indices to work with cluster files
inDtoFold = inDtoFold+1;
FoldInto = FoldInto+1;

%% Folding
for i = 1:size(inDtoFold,2)
    inpFile(FoldInto(i),:) = inpFile(inDtoFold(i),:)+inpFile(FoldInto(i),:);
end

%% Clear inDtoFold rows from inpFile
inpFile(inDtoFold',:) = [];

end