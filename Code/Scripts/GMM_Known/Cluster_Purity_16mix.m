
function [ind] = Cluster_Purity_16mix(SeqHMMLabel,SeqOrigLabel,NHmm,iternum)
% Function to evaluate cluster purity
% Parameters usage:
% SeqHMMLabel: cluster labels of all sequences 
% SeqOrigLabe: original labels of all sequences
% NHmm: total number of HMMs
% iternum: iteration number

    Map = zeros(size(unique(SeqOrigLabel),1),NHmm); % Confusion matrix initialization: Number of discovered
                                                    % HMMs times true number of HMMs
      
    for i = 1:size(SeqHMMLabel,1)
        Map(SeqOrigLabel(i),SeqHMMLabel(i)) = Map(SeqOrigLabel(i),SeqHMMLabel(i)) + 1;
    end
    
    ClassPurity = max(Map,[],1)./sum(Map,1);  % cluster purity for each cluster
    Overall_purity = sum(max(Map,[],1))/size(SeqOrigLabel,1);  % Overall cluster purity
    
    [~,ind] = max(Map,[],1);  % Mapping of true label for every cluster by max assignment
    
    folder = [pwd '/Cluster_Purity_16mix'];
    cd(folder);
    
    dlmwrite(['cluster_it_' num2str(iternum)],ClassPurity);
    dlmwrite(['cluster_it_' num2str(iternum)],Overall_purity,'-append');
    dlmwrite(['cluster_it_' num2str(iternum)],Map,'-append');
    
    cd('../');
    
end
