% Script for judging segmentation quality
% in terms of recall and oversegmentation

clear;

addpath([ pwd '/../../Database']);
addpath([ pwd '/../../Scripts/Unsegmented/Segmentation']);

% load the boundary variable file that contains the boundary variables for each TIMIT recording
load boundvars.mat; % This changes for scenario one wishes to test

load AbsAudioBound.mat;
load phone_bound_inds.mat;

CumsumindsApp = zeros(1,1);
CumsumindsApp = cat(2,CumsumindsApp,Cumsuminds');

OverSeg = zeros(size(AbsAudioBoundaryIndApp,2)-1,1);
Pc = zeros(size(AbsAudioBoundaryIndApp,2)-1,1);

for k = 1:size(AbsAudioBoundaryIndApp,2)-1
    
    startind = find(CumsumindsApp == AbsAudioBoundaryIndApp(k))+1;
    endind = find(CumsumindsApp == AbsAudioBoundaryIndApp(k+1));
    
    offset = CumsumindsApp(startind-1);
       
    [OverSegout,Pcout] = segment_quality(BoundVars{k,:},CumsumindsApp(startind:endind),offset);
    OverSeg(k) = OverSegout;
    Pc(k) = Pcout;
end

% Mean oversegmentation and recall (Pc)
Oversegmean = mean(OverSeg);
Pcmean = mean(Pc);
