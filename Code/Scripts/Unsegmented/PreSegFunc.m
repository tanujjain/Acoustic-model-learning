
function [BoundVars] = PreSegFunc(X_Unseg,AbsAudioBoundaryIndApp)
% Function to implement presegmentation algorithm
% where,
%   X_Unseg: Matrix containing all samples from all recordings. (ObsDim * total_no_of_samples_in_all_recordings)
%   AbsAudioBoundaryIndApp: Indices of the last sample in a TIMIT recording. (1*4621) vector, explained in readme document
%   BoundVars: Boundary variables for each recording, (4620*1) cell

    load phone_bound_inds.mat; % Get the original indices of the last sample in each phoneme, gives
                                % a vector called 'Cumsuminds'
    CumsumindsApp = zeros(1,1);
    CumsumindsApp = cat(2,CumsumindsApp,Cumsuminds'); % Cumsuminds contain true phoneme boundaries, required only for performance 
                                                      % analysis of presegmentation, CumsumindsApp adds a zero as the first element
                                                      % and changes shape of Cumsuminds for manipulation purposes
                                                    
    BoundVars = cell(1);  % To contain boundary variables for each TIMIT recording
    
    for k = 1:size(AbsAudioBoundaryIndApp,2)-1
        start = AbsAudioBoundaryIndApp(k)+1; % To get start and end indices for each recording
        last =  AbsAudioBoundaryIndApp(k+1);
   
        startind = find(CumsumindsApp == AbsAudioBoundaryIndApp(k))+1; % Get true boundaries for recording k
        endind = find(CumsumindsApp == AbsAudioBoundaryIndApp(k+1));
    
        offset = CumsumindsApp(startind-1); % index of the last sample in the previous recording, to be subtratced to remove the cumsum impact on indices

    
        [boundary,~,~] = PreSegment_2(X_Unseg(:,start:last),CumsumindsApp(startind:endind),offset); % Call presegmentation function
        BoundVars{k,:} = boundary;
    
    end

end