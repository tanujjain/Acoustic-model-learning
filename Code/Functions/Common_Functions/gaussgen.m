
function [out] = gaussgen(m,C,numsample)
% Function to generate gaussian random variables from a multivariate gaussian
% where:
%   m: mean of the gaussian distribution. (ObsDim*1) vector
%   C: variance of the gaussian distribution. (ObsDim*ObsDim) diagonal matrix 
%   numsample: number of samples to be generated

    dim=size(m,1);
    In= randn(dim,numsample); % Sample from standard normal
    [V,D]=eig(C);  % Eigen value decomposition of covariance matrix
    
    out = V*D^(1/2)*In + m*ones(1,numsample); 

end