%% Script to plot a confusion matrix

% Add path to the cluster purity folder where the cluster purity file is located
%<Add path here>

clear
close

ConfMat = dlmread('cluster_it_1'); % Read the cluster purity file whose confusion matrix is to be plotted
ConfMat = ConfMat(3:end,:);  % Remove first two rows, confusion matrix starts from 3rd lien of these files
ConfMat = ConfMat';

[count, phoneme] = max(ConfMat,[],2);
[phoneme_sorted, clusteridx] = sort(phoneme); % Sort phoneme labels for plotting
ConfMat = ConfMat(clusteridx, :);

%%
figure(1338)

for i = 1:size(ConfMat,2)
    ind = find(ConfMat(:,i)>200);
    plot(i*ones(1,size(ind,1)),ind','bo');
    hold on
end
xlabel('True phonemes lables');
ylabel('Discovered HMM cluster labels')
