/* Function to evaluate the observation probabiltiy for the sample-state pair for a given sequence-state lattice
 * and calculate the difference between the observation probability of each mixture and the total observation probabiltiy
 * of the sample for a state
 * Usage:
 *  [B,diff] = State_Obs_Prob_GMM(X,mu,cov,Nstates,Pi_Gmm)
 *  where:
 *      B: return attribute, contains the observation probabiltity values for each sample-state combination
 *      diff: Log(b for a mixture)-sum(b for all mixtures): for all samples and state
 *      X: sequence of samples. (ObsDim*sequence_length) matrix
 *      mu: means of all states and mixture components of the GMM for each state for this HMM.(ObsDim*Nmix*Nstates) matrix         
 *      cov: covariances of all states and mixture components of the GMM for each state for this HMM.(ObsDim*Nmix*Nstates) matrix
 *      Pi_Gmm: (Nmix*Nstates) matrix containing means for this HMM   
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
             
     const double* X = mxGetPr(prhs[0]);
     const double* mu = mxGetPr(prhs[1]);
     const double* cov = mxGetPr(prhs[2]);
     const double* PiGmm = mxGetPr(prhs[4]);
       
     const mwSize dims[]={Nstates,Nsamples,Nmix+1};
     const mwSize dimsDiff[]={Nstates,Nsamples,Nmix};

     plhs[0] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
     double* B = mxGetPr(plhs[0]);             
     plhs[1] = mxCreateNumericArray(3,dimsDiff,mxDOUBLE_CLASS,mxREAL);
     double* Diff = mxGetPr(plhs[1]);
     
     double right;
     double result;
     
     for(unsigned int i=0;i<Nstates;++i){
         for(unsigned int j=0;j<Nsamples;++j){
                     
             double *sums = (double *)malloc(sizeof(double)*Nmix);
             for(unsigned int m=0;m<Nmix;++m){
                right = 0;
                for(unsigned int k=0;k<ObsDim;++k)                                                 
                    right = right + log((cov[i*ObsDim*Nmix+(ObsDim*m+k)])) + 
                            (pow(((X[ObsDim*j+k]) - (mu[i*ObsDim*Nmix+(ObsDim*m+k)])),2)/(cov[i*ObsDim*Nmix+(ObsDim*m+k)])); // log(var(dim,i))+Exponential Part of Gaussian
                                        
                 sums[m] = log(PiGmm[i*Nmix+m])-(ObsDim/2.0)*log(2*3.1416)-0.5*right; // (-D/2)*log(2*pi)-(1/2)*right
                 B[Nstates*Nsamples*m+Nstates*j+i] = sums[m];
                     
             }
             result = sums[0];
             for(unsigned int mix=0;mix<Nmix-1;++mix)
                 result = LogAdd(result, sums[mix+1]);
                     
             B[Nstates*Nsamples*Nmix+Nstates*j+i] = result;
             free(sums);
                     
                 }
             } 
     
       for(unsigned int i=0;i<Nstates;++i)
           for(unsigned int j=0;j<Nsamples;++j)
               for(unsigned int m=0;m<Nmix;++m)
                   Diff[Nstates*Nsamples*m+Nstates*j+i] = B[Nstates*Nsamples*m+Nstates*j+i] - B[Nstates*Nsamples*Nmix+Nstates*j+i];

        
}