/*
   * Function to calculate the backward variable for a sequence given the HMM parameters
   * Usage: Beta = BackwardAlgorithmLog(B,log(Aest),log(Pi))
   * where:
   *        B: Observation probabilities for this sequence. (Nstates*sequence_length) matrix
   *        log(Aest): log of the transition amtrix for this HMM
   *        NHmm : Total number of HMMs
   *        log(Pi): log ofInitial State Probabilities
   *        Beta: return backward variable (Nstates*sequence_length) matrix 
 */

#include "mex.h"
#include <cmath>
#include <limits>
#include <math.h>
#include <boost/math/special_functions/log1p.hpp>
  using boost::math::log1p;


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


void mexFunction(int nlhs,       mxArray *plhs[ ],
                 int nrhs, const mxArray *prhs[ ]) {
    const double* G = mxGetPr(prhs[0]);
    const double* T = mxGetPr(prhs[1]);
    unsigned int NumStates = mxGetN(prhs[1]);
    unsigned int NumObs = mxGetN(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(NumStates, NumObs, mxREAL); //beta
    
    double* beta = mxGetPr(plhs[0]);
    
    //Initialisieren
    if(nrhs<3){
        //Kein Anfangszustand gegeben
        for(unsigned int IdxState=0; IdxState<NumStates; IdxState++)
            beta[IdxState+NumStates*(NumObs-1)] = 0;
    }
    else{
        //Anfangszustand gegeben
        for(unsigned int IdxState=0; IdxState<NumStates; IdxState++)
            beta[IdxState+NumStates*(NumObs-1)] = (mxGetPr(prhs[2]))[IdxState];
    }
    
    //Backward Algorithmus
    for(unsigned int IdxObs=NumObs-1; IdxObs>0; IdxObs--){
        for(unsigned int IdxState1=0; IdxState1 < NumStates; IdxState1++){
            double logsum=T[IdxState1] + G[(NumStates*(IdxObs))] + beta[(NumStates*(IdxObs))];
            for(unsigned int IdxState2=1; IdxState2 < NumStates; IdxState2++){
                /*mexPrintf("Idx State1: %d\n", IdxState1);
                mexPrintf("Idx State2: %d\n", IdxState2);
                mexPrintf("IdxObs: %d\n\n", IdxObs);*/
                logsum = LogAdd(logsum, T[IdxState1+NumStates*IdxState2] + G[IdxState2+(NumStates*(IdxObs))] + beta[IdxState2+(NumStates*(IdxObs))]);
            }
            beta[IdxState1+NumStates*(IdxObs-1)] = logsum;
        }
    }
}