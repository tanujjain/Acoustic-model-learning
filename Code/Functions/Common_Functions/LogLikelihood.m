function [result]= LogLikelihood(x,m,sig)

%Calculate Loglikeihood of sample x to belong to a gaussian with mean m and
%covariance sig

dim=size(x,1);
bracket= ((x-m).^2)';
bracket=bracket*(1./sig);
bracket=-0.5*bracket;

c=((2*pi)^(dim/2))*sqrt(prod(sig));
result = log(1/c) + bracket;


% expval=exp(bracket);
% 
% result=((2*pi)^(dim/2))*sqrt(prod(sig));
% 
% result=(1/result)*expval;
% 
% result=log(result);

end
