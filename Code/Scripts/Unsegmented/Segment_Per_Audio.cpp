
/*
   * Function to retrieve phoneme boundaries in an audio recording sequence 
   * Usage: boundary_vector = Segment_Per_Audio(X_Unseg,meanmat,covmat,Pigmmmat,Total_Pi,Total_trans_mat);
   * where:
   *        X_Unseg : All Data in the given audio recording
   *        meanmat : mean matrix of all states and mixtures with dimensions ObsDim*Nmix*NumStates
   *        covmat : cov matrix of all states and mixtures with dimensions Nmix*NumStates
   *        Pigmmmat : Pi_Gmm matrix of all states and mixtures with dimensions ObsDim*Nmix*NumStates
   *        Total_Pi : Initial State Probabilities for all HMMs considered as one huge HMM
   *        Total_trans_mat : Total transition matrix of all HMM states considered together
   *        boundary_vector : ouput boundary vector having dimensions 1*NumObs with a '0' for no boudnary and a '1' if boundary present
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
  
  
// Calculate the State Observation Probabilities  
double* State_Prob(double *B,double *tmp_X,double *tmp_mu,double *tmp_cov,double *tmp_PiGmm,
                    int ObsDim,int nsamples,int nstates,int Nmix){    
    double right;
    double result;
         for(unsigned int i=0;i<nstates;++i){
                 for(unsigned int j=0;j<nsamples;++j){
                     double *sums = (double *)malloc(sizeof(double)*Nmix);
                     for(unsigned int m=0;m<Nmix;++m){
                        right = 0;
                        for(unsigned int k=0;k<ObsDim;++k)                                                 
                            right = right + log((tmp_cov[i*ObsDim*Nmix+(ObsDim*m+k)])) + 
                                 (pow(((tmp_X[ObsDim*j+k]) - (tmp_mu[i*ObsDim*Nmix+(ObsDim*m+k)])),2)/(tmp_cov[i*ObsDim*Nmix+(ObsDim*m+k)])); // log(var(dim,i))+Exponential Part of Gaussian
                                        
                        sums[m] = log(tmp_PiGmm[i*Nmix+m])-(ObsDim/2.0)*log(2*3.1416)-0.5*right; // (-D/2)*log(2*pi)-(1/2)*right
                     }
                     result = sums[0];
                     for(unsigned int mix=0;mix<Nmix-1;++mix)
                       result = LogAdd(result, sums[mix+1]);
                     
                     B[nstates*j+i] = result;
                     free(sums);
                 }
             }
        return B;
        
}

// Display the calculated values of B (State Observation Probabilities) --> For debugging purposes only
void disp_B(double *B,int len){
    for(unsigned int i=0;i<len;++i){
        mexPrintf("\n B val = %f",B[i]);
    }
}

// Find the state with maximum probability of occurrence for the last sample in the sequence (to get the starting point for backtracking)
int Findmax(double *arr,int Nelements){
    int maxind = 0;
            for(unsigned int index=1;index<Nelements;++index)
                if(arr[index] > arr[index-1]){
                    maxind = index;
                }
            return maxind;
} 


int* Viterbi(int* State,double *B,double *tmp_Aest,double *Pi,int NumObs,int NumStates){
    
    double *alpha = (double *)malloc(sizeof(double)*(NumStates*2));
     for(unsigned int IdxState=0; IdxState<NumStates; IdxState++)
            alpha[IdxState] = B[IdxState] + log(Pi[IdxState]);      // log(b_{IdxState}(O_{1})) + log(Pi(IdxState))
     
    int MaxTrack[(NumObs-1)*NumStates]; // To keep track of maximum state from previous observation

    //Forward Algorithm
    for(unsigned int IdxObs=1; IdxObs<NumObs; IdxObs++){                                       // Goes over all observations (1 <= IdxObs < T)
        for(unsigned int IdxState2=0; IdxState2 < NumStates; IdxState2++){                     // Goes over current Observation (0 <= IdxState2 < NumStates)
            double SumVec[NumStates];
            
            for(unsigned int IdxState1=0; IdxState1 < NumStates; IdxState1++)               // Goes over all alphas of last observation (0 <= IdxState1 < NumStates)
                   SumVec[IdxState1] = log(tmp_Aest[IdxState1+NumStates*IdxState2]) + alpha[IdxState1]; 
            
            int maxind = Findmax(SumVec,NumStates);
            
            MaxTrack[IdxState2+NumStates*(IdxObs-1)] = maxind;
            alpha[IdxState2+NumStates] = SumVec[maxind] + B[IdxState2+NumStates*IdxObs];        // Weighing with current observation state probability
        }
        
        for(int id = 0;id < NumStates;id++){
            alpha[id] = alpha[NumStates+id];
        }
    }

    // Backtrack Algo
    int trackind;
    
    State[NumObs-1] = Findmax(&alpha[NumStates],NumStates);
    for(int ind = NumObs-2;ind > -1;ind--){
        trackind = State[ind+1];
        State[ind] = MaxTrack[NumStates*ind+trackind];
    }

    free(alpha);
    return State;    
}

// Gets boundaries by detecting the change in states from one HMM to another
void GetBoundaries(int* BoundaryVec,int* StateVec,int NumObs){
    for(unsigned int i=0;i<NumObs-1;++i){
        if(StateVec[i]%3 == 0 && StateVec[i+1]%3 == 1)
            BoundaryVec[i] = 1;
        else
            BoundaryVec[i] = 0;
    }
    BoundaryVec[NumObs-1] = 1;
}

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{                
    double *tmp_X = mxGetPr(prhs[0]);
    const int* dims_X = mxGetDimensions(prhs[0]);// Get dimensions of Observation matrix 
        
    double* tmp_mu = mxGetPr(prhs[1]);
    double* tmp_cov = mxGetPr(prhs[2]);
    double* tmp_PiGmm = mxGetPr(prhs[3]);
    double *Pi = mxGetPr(prhs[4]);
    double* tmp_Aest = mxGetPr(prhs[5]);
    
    const int *dims_Aest = mxGetDimensions(prhs[5]); // Get Dimensions of Aest
    const int *dims_PiGmm = mxGetDimensions(prhs[3]); // Get Dimesnions of Pi_Gmm
    

    int *state = (int *)malloc(sizeof(int)*dims_X[1]);

    plhs[0] = mxCreateNumericMatrix(1, dims_X[1], mxUINT32_CLASS, mxREAL); // return boundary variables
    int* boundaries   = (int*) mxGetData(plhs[0]);
          
    double *B = (double *)malloc(sizeof(double)*dims_Aest[0]*dims_X[1]);  // Observation probabilities
    B = State_Prob(B,tmp_X,tmp_mu,tmp_cov,tmp_PiGmm,dims_X[0],dims_X[1],dims_Aest[0],dims_PiGmm[0]);          
    state = Viterbi(state,B,tmp_Aest,Pi,dims_X[1],dims_Aest[0]); // Viterbi algorithm to assign states from cascaded state space
    
    for (unsigned int i=0;i<dims_X[1];++i){
        state[i] = state[i]+1;
    }
    
    GetBoundaries(boundaries,state,dims_X[1]);
   
    free(B);
    free(state);  
}                   
