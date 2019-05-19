
function [OverSeg,Pc] = segment_quality(DiscoveredSegBounds_total,OrigBoundaryInd,offset)
% Function to get segmentation quality of one sequence
% where,
%   DiscoveredSegBounds_total: Discovered boundaries. Vector (1*no_of_samples_in_recording)
%   OrigBoundaryInd: Index of original boundaries for this sequence. Vector (1*no_of_boundaries_in_this_seq)
%   offset: Offset to counter a zero at the beginning of vector OrigBoundaryInd. Scalar

OrigBoundaryInd = OrigBoundaryInd-offset;
DiscoveredSegBounds = find(DiscoveredSegBounds_total>0);

OverSeg = size(find(DiscoveredSegBounds>0),2)/size(OrigBoundaryInd,2);

CorrectDetect = 0;
for index = 1:size(OrigBoundaryInd,2)
    if (sum(DiscoveredSegBounds == OrigBoundaryInd(index))+...
            sum(DiscoveredSegBounds+2 == OrigBoundaryInd(index))+...
            sum(DiscoveredSegBounds-2 == OrigBoundaryInd(index)))~=0
        CorrectDetect = CorrectDetect+1;
    end
end

Pc = CorrectDetect/size(OrigBoundaryInd,2);
end