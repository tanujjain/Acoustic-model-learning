/* Function to evaluate the observation probabiltiy for the sample-state pair for a given sequence-state lattice
 * Usage:
 *  B = State_Obs_Prob_GMM(X,mu,cov,Nstates,Pi_Gmm)
 *  where:
 *      B: return attribute, contains the observation probabiltity values for each sample-state combination
 *      X: sequence of samples
 *      mu: means of all states and mixture components of the GMM for each state         
 *      cov: covariances of all states and mixture components of the GMM for each state
 *      Pi_Gmm: weights of each GMM mixture component for each state    
 */

#include "mex.h"
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <boost/math/special_functions/log1p.hpp>
  using boost::math::log1p;  

  
// Logadd
double LogAdd(double a, double b) {
    double c;
    
    if(a > b) {
        c = b;
        b = a;
        a = c;
    }
    c = a - b;
    
    if((a == -std::numeric_limits<double>::infinity()) || (c < -36)) {
        return(b);
    } else {
        return(b + log1p(exp(c)));
    }
}  


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]) {
                        
     unsigned int Nstates = mxGetScalar(prhs[3]);
     unsigned int Nsamples = mxGetN(prhs[0]);
     unsigned int ObsDim = mxGetM(prhs[0]);
     unsigned int Nmix = mxGetM(prhs[4]);
             
     const double* X = mxGetPr(prhs[0]); // Get pointers to each input parameter
     const double* mu = mxGetPr(prhs[1]);
     const double* cov = mxGetPr(prhs[2]);
     const double* PiGmm = mxGetPr(prhs[4]);
             
     plhs[0] = mxCreateDoubleMatrix(Nstates, Nsamples, mxREAL);
     double* B = mxGetPr(plhs[0]); // Set B as output            
    
     double right;
     double result;
         for(unsigned int i=0;i<Nstates;++i){ // Goes over each state
             for(unsigned int j=0;j<Nsamples;++j){ // Goes over each sample                    
                 double *sums = (double *)malloc(sizeof(double)*Nmix);
                 for(unsigned int m=0;m<Nmix;++m){ // Goes over each mixture
                    right = 0;
                    for(unsigned int k=0;k<ObsDim;++k) // Goes over each observation dimension                                                
                        right = right + log((cov[i*ObsDim*Nmix+(ObsDim*m+k)])) + 
                                 (pow(((X[ObsDim*j+k]) - (mu[i*ObsDim*Nmix+(ObsDim*m+k)])),2)/(cov[i*ObsDim*Nmix+(ObsDim*m+k)])); // log(var(dim,i))+Exponential Part of Gaussian
                                        
                    sums[m] = log(PiGmm[i*Nmix+m])-(ObsDim/2.0)*log(2*3.1416)-0.5*right; // (-D/2)*log(2*pi)-(1/2)*right
                 }
                    result = sums[0];
                    for(unsigned int mix=0;mix<Nmix-1;++mix)
                        result = LogAdd(result, sums[mix+1]);
                     
                    B[Nstates*j+i] = result;
                    free(sums);
                 }
             }               
}