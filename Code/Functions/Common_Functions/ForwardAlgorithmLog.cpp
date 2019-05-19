/*
   * Function to calculate the forward variable for a sequence given the HMM parameters
   * Usage: Alpha = ForwardAlgorithmLog(B,log(Aest),log(Pi))
   * where:
   *        B: Observation probabilities for this sequence. (Nstates*sequence_length) matrix
   *        log(Aest): log of the transition amtrix for this HMM
   *        NHmm : Total number of HMMs
   *        log(Pi): log ofInitial State Probabilities
   *        Alpha: return forward variable (Nstates*sequence_length) matrix 
 */


#include "mex.h"
#include <cmath>
#include <limits>
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
    const double* G = mxGetPr(prhs[0]);                      // Gewichtung im log-space (Beobachtungswsk log(b_{i}(O_{t})), 0 <= i < NumStates, 0 <= t < T)
    const double* T = mxGetPr(prhs[1]);                      // Transitionsmatrix im log-space
    unsigned int NumStates = mxGetN(prhs[1]);                // Anzahl der States (Anzahl der Worte mal Anzahl der moeglichen Token pro Wort)
    unsigned int NumObs = mxGetN(prhs[0]);                   // Anzahl der Beobachtung (Laenge der Ausserung)
    
    
    // Initialisierung der Ausgabe
    plhs[0] = mxCreateDoubleMatrix(NumStates, NumObs, mxREAL); 	// alpha
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL); 			// P_O
    
    double* alpha = mxGetPr(plhs[0]);
    double* P_O   = mxGetPr(plhs[1]);
    
    //Initialisieren
    if(nrhs<3){
        //Kein Anfangszustand gegeben
        for(unsigned int IdxState=0; IdxState<NumStates; IdxState++)
            alpha[IdxState] = G[IdxState] - log((double)NumStates);
    }
    else{
        //Anfangszustand gegeben
        for(unsigned int IdxState=0; IdxState<NumStates; IdxState++)
            alpha[IdxState] = G[IdxState] + (mxGetPr(prhs[2]))[IdxState];      // log(b_{IdxState}(O_{1})) + log(Pi(IdxState))
    }
    
    //Forward Algorithmus
    for(unsigned int IdxObs=1; IdxObs<NumObs; IdxObs++){                                       // Gehe ueber alle Beobachtungen (1 <= IdxObs < T)
        for(unsigned int IdxState2=0; IdxState2 < NumStates; IdxState2++){                     // Gehe ueber alle alpha zur aktuellen Beobachtung (0 <= IdxState2 < NumStates)
            double logsum=T[NumStates*IdxState2] + alpha[NumStates*(IdxObs-1)];
            for(unsigned int IdxState1=1; IdxState1 < NumStates; IdxState1++){                 // Gehe ueber alle alpha der letzten Beobachtung (0 <= IdxState1 < NumStates)
                /*mexPrintf("Idx State1: %d\n", IdxState1);
                mexPrintf("Idx State2: %d\n", IdxState2);
                mexPrintf("IdxObs: %d\n", IdxObs);
                mexPrintf("Index T: %d\n1 IdxState1+NumStates*IdxState2);
                mexPrintf("Index alpha: %d\n\n", IdxState1+NumStates*(IdxObs-1));*/
                logsum = LogAdd(logsum, T[IdxState1+NumStates*IdxState2] + alpha[IdxState1+NumStates*(IdxObs-1)]);
            }
            alpha[IdxState2+NumStates*IdxObs] = logsum + G[IdxState2+NumStates*IdxObs];        // Gewichtet mit der aktuellen Beobachtung
        }
    }
    double alphaLogSum = alpha[0+NumStates*(NumObs-1)];
    for(unsigned int IdxState1=1; IdxState1 < NumStates; IdxState1++){
        alphaLogSum = LogAdd(alphaLogSum, alpha[IdxState1+NumStates*(NumObs-1)]);
    }
    P_O[0]=alphaLogSum;//P_O
}