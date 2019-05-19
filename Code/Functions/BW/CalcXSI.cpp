/* Function to evalaute the Xsi paramter of the Baum-Welch algorithm (terminology borrowed from Rabiner's tutorial on HMM)
 * Usage:
 * logxsi = CalcXSI(alpha,B+Beta,log(Aest))   []
 * where:
 *  alpha: forward variable Forward-Backward algorithm for all samples in the sequence except last
 *  B: Observation probability for a sample-state pair
 *  Beta: backward variable of the Forward-Backward algorithm
 **/
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
                        
     unsigned int Nstates = mxGetM(prhs[0]);
     unsigned int Nsamples = mxGetN(prhs[0]);
     const double* Alpha = mxGetPr(prhs[0]); // get pointer to inputs
     const double* BplusBeta = mxGetPr(prhs[1]);
     const double* Aest = mxGetPr(prhs[2]);
     
     const mwSize dims[]={Nstates,Nstates,Nsamples};
     plhs[0] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
     double* xsi = mxGetPr(plhs[0]); // get pointer to output
     
     for(unsigned int i=0;i<Nsamples;++i)
         for(unsigned int j=0;j<Nstates;++j)
             for(unsigned int k=0;k<Nstates;++k)
                 xsi[Nstates*Nstates*i+Nstates*j+k] = Alpha[Nstates*i+k]+BplusBeta[Nstates*i+j]+  //refer formula from Rabiner paper
                                                                            Aest[Nstates*j+k];
                     
     for(unsigned int i=0;i<Nsamples;++i){
         double sums = -std::numeric_limits<double>::infinity();
         for(unsigned int j=0;j<Nstates;++j)
             for(unsigned int k=0;k<Nstates;++k)
                 sums = LogAdd(xsi[Nstates*Nstates*i+Nstates*j+k],sums);

         for(unsigned int j=0;j<Nstates;++j)
                 for(unsigned int k=0;k<Nstates;++k)
                     xsi[Nstates*Nstates*i+Nstates*j+k] = xsi[Nstates*Nstates*i+Nstates*j+k]-sums;                                          
     }                     
                                         
   }