function [result]= Likelihood(x,m,sig)
%Calculate likeihood of sample x to belong to a gaussian with mean m and
%covariance sig

dim=size(x,1);
bracket= ((x-m).^2)';
bracket=bracket*(1./sig);
bracket=-0.5*bracket;

expval=exp(bracket);

result=((2*pi)^(dim/2))*sqrt(prod(sig));

result=(1/result)*expval;

end