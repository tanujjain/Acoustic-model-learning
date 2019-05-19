function [d]=dirichlet_gen(betas)
% Function to sample from Dirichlet Distribution
% where:
%   betas: vector containing Dirichlet Hyperparameters
%   d: vector with same size as that of betas conatining the samples

d=gamrnd(betas,1);
d=d./(sum(d,2));
end