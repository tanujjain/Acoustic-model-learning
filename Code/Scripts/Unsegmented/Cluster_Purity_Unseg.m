
function [ind] = Cluster_Purity_Unseg(SeqHMMLabel,SeqOrigLabel,NHmm,iternum)
% Function to evaluate cluster purity
% Parameters usage:
% SeqHMMLabel: cluster labels of all sequences 
% SeqOrigLabe: original labels of all sequences
% NHmm: total number of HMMs
% iternum: iteration number
%   return:
%       ind: indices of HMM to which each cluster is assigned thorugh max assignment (original labels of majority number of sequences in each cluster) 

    Map = zeros(48,NHmm); % Confusion matrix initialization: Number of discovered
                          % HMMs times true number of HMM
    
    for i = 1:size(SeqHMMLabel,1)
        Map(SeqOrigLabel(i),SeqHMMLabel(i)) = Map(SeqOrigLabel(i),SeqHMMLabel(i)) + 1;
    end
    
    ClassPurity = max(Map,[],1)./sum(Map,1);  % cluster purity for each cluster
    Overall_purity = sum(max(Map,[],1))/sum(Map(:)); % Overall cluster purity
    
    [~,ind] = max(Map,[],1);  % Mapping of true label for every cluster by max assignment
    
    folder = [pwd '/Cluster_Purity'];
    cd(folder);
    
    dlmwrite(['cluster_it_' num2str(iternum)],ClassPurity);

    dlmwrite(['cluster_it_' num2str(iternum)],Overall_purity,'-append');

    dlmwrite(['cluster_it_' num2str(iternum)],Map,'-append');
    
    cd('../');
    
end
