
/*
   * Function to calculate the likelihood of the sequence given the model HMMs
   * Usage: Probs = HMMLikelihoodCalC(index,NHmm,Pi,X,Aest,mu,cov,Pi_Gmm);
   * where:
   * where:
   *        i: index of the sequence for this this evaluation needs to be done
   *        NHmm: Total number of HMMs in the system
   *        Pi: Initial state probability for all HMMs. vector with value = [1 0 0]
   *        X: all sequences. (num_sequences*1) cell with each cell containing a sequence. Each cell entry is (ObsDim*sequence_length) matrix
   *        Aest: Transition matrices for all HMMs. (NHmm*1) cell with each cell entry a (Nstates*Nstates) matrix 
   *        mu: means for all HMMs. (NHmm*1) cell with each cell entry a (ObsDim*Nmix*Nstates) matrix
   *        cov: covariances for all HMMs. (NHmm*1) cell with each cell entry a (ObsDim*Nmix*Nstates) matrix
   *        Pi_Gmm: weights of each Gaussian in the GMM for all states and HMMs. (NHmm*1) cell which each entry as (Nmix*Nstates) matrix
   *        Probs:  (1*NHmm) vector containing likelihood of a sequence to belong to each of the NHmm HMMs
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


// Calculates Forward Variable and returns its sum to give the likelihood of a sequence given state observation probabilities & Transition probabilities
double Calc_Forward(double *B,double *tmp_Aest,double *Pi,int NumObs,int NumStates){
    
    double *alpha = (double *)malloc(sizeof(double)*NumStates*NumObs);
     for(unsigned int IdxState=0; IdxState<NumStates; IdxState++)
            alpha[IdxState] = B[IdxState] + log(Pi[IdxState]);      // log(b_{IdxState}(O_{1})) + log(Pi(IdxState))
    

    //Forward Algorithm
    for(unsigned int IdxObs=1; IdxObs<NumObs; IdxObs++){                                       // Goes over all observations (1 <= IdxObs < T)
        for(unsigned int IdxState2=0; IdxState2 < NumStates; IdxState2++){                     // Goes over current Observation (0 <= IdxState2 < NumStates)
            double logsum=log(tmp_Aest[NumStates*IdxState2]) + alpha[NumStates*(IdxObs-1)];
            for(unsigned int IdxState1=1; IdxState1 < NumStates; IdxState1++){                 // Goes over all alphas of last observation (0 <= IdxState1 < NumStates)
                logsum = LogAdd(logsum, log(tmp_Aest[IdxState1+NumStates*IdxState2]) + alpha[IdxState1+NumStates*(IdxObs-1)]);
            }
            alpha[IdxState2+NumStates*IdxObs] = logsum + B[IdxState2+NumStates*IdxObs];        // Weighing with current observation state probability
        }
    }

    double alphaLogSum = alpha[0+NumStates*(NumObs-1)];    // Sum of alpha for last observation of the sequence
    for(unsigned int IdxState1=1; IdxState1 < NumStates; IdxState1++){
        alphaLogSum = LogAdd(alphaLogSum, alpha[IdxState1+NumStates*(NumObs-1)]);
    }
    
    free(alpha);
    return alphaLogSum;    // return the sum of alphas of last observation
}


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{                
    mxArray *cell_element_ptr_X,*cell_element_ptr_Aest,*cell_element_ptr_mu,
            *cell_element_ptr_cov,*cell_element_ptr_PiGmm;
    
    unsigned int index = mxGetScalar(prhs[0]);
    unsigned int nhmm = mxGetScalar(prhs[1]);
    
    plhs[0] = mxCreateDoubleMatrix(1,nhmm,mxREAL); 
    double* Alpha_Sum   = mxGetPr(plhs[0]);
    double *Pi = mxGetPr(prhs[2]);
      
    cell_element_ptr_X = mxGetCell(prhs[3],index);      // Get Pointer to exact sequence indexed by index variable within cell
    const int* dims_X = mxGetDimensions(cell_element_ptr_X);// Get dimensions of Observation cell 
    double* tmp_X = (double *)mxGetData(cell_element_ptr_X);// Get pointer to elements within sequence
    
    cell_element_ptr_Aest = mxGetCell(prhs[4],1);     
    const int *dims_Aest = mxGetDimensions(cell_element_ptr_Aest); // Get Dimensions of Aest
    
    cell_element_ptr_PiGmm = mxGetCell(prhs[7],1);     
    const int *dims_PiGmm = mxGetDimensions(cell_element_ptr_PiGmm);
    
    for(unsigned int i=0;i<nhmm;++i){       // Goes over all HMMs
       cell_element_ptr_Aest = mxGetCell(prhs[4],i); // Get pointers to relevant HMM component of the cell
       cell_element_ptr_mu = mxGetCell(prhs[5],i);
       cell_element_ptr_cov = mxGetCell(prhs[6],i);
       cell_element_ptr_PiGmm = mxGetCell(prhs[7],i);
       
       double* tmp_Aest = (double *)mxGetData(cell_element_ptr_Aest);
       double* tmp_mu = (double *)mxGetData(cell_element_ptr_mu);
       double* tmp_cov = (double *)mxGetData(cell_element_ptr_cov);
       double* tmp_PiGmm = (double *)mxGetData(cell_element_ptr_PiGmm);
       
       double *B = (double *)malloc(sizeof(double)*dims_Aest[0]*dims_X[1]);  // Observatio  probabilities
       B = State_Prob(B,tmp_X,tmp_mu,tmp_cov,tmp_PiGmm,dims_X[0],dims_X[1],dims_Aest[0],dims_PiGmm[0]);          
       Alpha_Sum[i] = Calc_Forward(B,tmp_Aest,Pi,dims_X[1],dims_Aest[0]);
       free(B);
    }                   
}
