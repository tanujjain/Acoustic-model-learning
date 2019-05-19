
function [cov] = Floor_cov(cov,Nmix,hmmnum)
% Function to floor covariance
% where,
%	cov: covariance matrix (ObsDim*Nmix), a column for each mixture since each mixture covaraince is diagonal
%	Nmix: number of Gaussian mixture components
%	hmmnum: the hmm number for which function is under execution

ObsDim = size(cov,1);
for i = 1:ObsDim
    for j = 1:Nmix
        if(cov(i,j)<0.01)
            cov(i,j) = 0.01;
            disp('floored =');disp(hmmnum);            
        end
    end
end

end
