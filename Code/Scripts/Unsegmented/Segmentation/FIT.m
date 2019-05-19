function [nwin] = FIT(S,p,q)
% Function to obtain the sample index that qualifies as a boundary 
% where:
%   S: Matix containing 1 for a candidate boundary points, 0 for others. (ObsDim*no_of_samples)
%   p,q: limits of the window within which a winning index needs to be decided
%   nwin: winning sample index

f = zeros(1,q-p+1);

    for i = 1:size(S,1) % Populate f
        for m = p:q        
            f(1,m-p+1) = f(1,m-p+1)+S(i,m)*abs(p-m);
        end
    end

    f = f(1,2:end);
    [~,nwin] = min(f); % get nwin as the index which is closes to a prospective boundary
    nwin = nwin+p;
end