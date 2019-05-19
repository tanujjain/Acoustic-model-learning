function [BoundVar,OverSeg,Pc] = PreSegment_2(X_unsegmented,OrigBoundaryInd,offset)
% Function to do presegmentation for a given sequence of samples
% where,
%   X_unsegmented: unsegmented sequence of samples, corresponds to a recording. Matrix (ObsDim*no_of_samples_in_a_recording)
%   OrigBoundaryInd: True indices in the sequences
%   offset: index of the last sample in the previous recording, to be subtratced to remove the cumsum impact on indices
%   BoundVar: boundary variable values, (1*no_of_samples) vector
%   OverSeg: Scalar, gives oversegmentation value
%   Pc: Scalar, gives recall
% The suggestions and terminology has been borrowed from the paper, 'A new text-independent method for phoneme segmentation.'
% by Guido Aversano, Anna Esposito, Antonietta Esposito, and Maria Marinaro.

OrigBoundaryInd = OrigBoundaryInd-offset; 

a = 10;  % number samples which precede and succeed the sample for which value of boundary variable is to be decided
% b = 2;  % Threshold for retaining peaks
c = 10;   % Window length to find dostance from a ürospective boundary, refer thesis report for exact explanation

ObsDim = 39;

N = size(X_unsegmented,2); % Number of Samples
X_unsegmented = normr(X_unsegmented);
J = zeros(39,N);

for n = a+1:N-a
    J(:,n) = sum(X_unsegmented(:,n-a:n-1),2)-sum(X_unsegmented(:,n+1:n+a),2); % determine jump function
end

J = J/a;
J = J(:,a+1:N-a);  % Retain columns within F'
% J = normr(J); % Normalize each row of J
for d = 1:size(J,1)
    range = max(J(d,:))-min(J(d,:));
    J(d,:) = (J(d,:)-min(J(d,:)))/range;
end

S = zeros(ObsDim,N);
for ii = 1:ObsDim
    pp = ones(1,size(J,2));
    PeakIndToTest = find(pp);
    
    Jtest = J(ii,:);
    LeftPeak = zeros(1,size(PeakIndToTest,2)); % The highest peak to the left and right of current sample
    RightPeak = zeros(1,size(PeakIndToTest,2));

    for i=1:size(PeakIndToTest,2)
        value = Jtest(PeakIndToTest(i));
        l = PeakIndToTest(i)-1;  % get index of the sample with amplitude greater than the last one.
    
        while(l>=1 && value > Jtest(l))
            value = Jtest(l);
            l = l-1;
        end
            if (l == PeakIndToTest(i)-1)
                LeftPeak(i) = 0;
                RightPeak(i) = 0;
            else
                LeftPeak(i) = l+1;
                m = PeakIndToTest(i)+1;
                value = Jtest(PeakIndToTest(i));
            
                while (m<=size(Jtest,2) && value >=Jtest(m))
                    value = Jtest(m);
                    m = m+1;
                end
            
                if m == PeakIndToTest(i)+1
                    LeftPeak(i) = 0;
                    RightPeak(i) = 0;
                else
                    RightPeak(i) = m-1;
                end
        
            end
    
    end

    Height = min(Jtest(LeftPeak~=0)-Jtest(LeftPeak(LeftPeak~=0)),Jtest(LeftPeak~=0)-Jtest(RightPeak(RightPeak~=0)));
    b = mean(J(ii,:),2); % Set threshold to the mean of the jump function value
    ThreshPeakInd = find(RightPeak);
    ThreshPeakInd2 = ThreshPeakInd(Height>b); % Retain peaks with height greater than b
    aa = S(ii,:);
    aa(ThreshPeakInd2)=1;
    S(ii,:)=aa;
end

acc = zeros(1,N); % accumulator vector

for nn = a:N-a-c
    nwin = FIT(S,nn,nn+c); % Apply FIT procedure to get a winner for each window
    acc(nwin) = acc(nwin)+1;
end

OverSeg = size(find(acc>1),2)/size(OrigBoundaryInd,2);
DiscoveredSegBounds = find(acc>1);

CorrectDetect = 0;
for index = 1:size(OrigBoundaryInd,2) % Boundary is correctly detected if it is within +-2 samples of true boundary
    if (sum(DiscoveredSegBounds == OrigBoundaryInd(index))+...
            sum(DiscoveredSegBounds+2 == OrigBoundaryInd(index))+...
            sum(DiscoveredSegBounds-2 == OrigBoundaryInd(index)))~=0
        CorrectDetect = CorrectDetect+1;
    end
end

Pc = CorrectDetect/size(OrigBoundaryInd,2); % Recall

%% Ensure min seq length is 3
FinalBound = ones(1,1);

for j = 2:size(DiscoveredSegBounds,2)
    if(DiscoveredSegBounds(j)-FinalBound(end) >= 3)
        FinalBound = cat(2,FinalBound,DiscoveredSegBounds(j));
    end
end

if(FinalBound(end)-DiscoveredSegBounds(end)< 3)
    FinalBound(end) = N;
else
    FinalBound = cat(2,FinalBound,N);
end

% Set initial boundary variables
BoundVar = zeros(1,N);
BoundVar(FinalBound) = 1;
BoundVar(1) = 0;

end
