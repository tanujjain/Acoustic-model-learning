
/*
   * Function to sample the parameters for a GMM of a given state only from the priors (Sample from the Base distribution)
   * Usage: [mu,cov,Pi_Gmm,Aest] = Sample_GMM(globmu,globcov,Nmix,Nstates);
   * where:
   *        Nmix: Total number of mixtures per GMM
   *        Nstates: Number of states per HMM
   *        Aest: Transition matrix. (Nstates*Nstates) matrix
   *        mu: means for each state. (ObsDim*Nmix) matrix
   *        cov: Diagonal of covariance matrix for each state. (ObsDim*Nmix) matrix
   *        Pi_Gmm: weights of each Gaussian in the mixture which represents this state. (Nmix*1) vector
   *        globmu: global mean of the database. (ObsDim*1) vector
   *        globcov: global covariance of the database. (ObsDim*1) vector
   */

#include "mex.h"
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/random/uniform_real.hpp>

// Generate Gaussian R.Vs with scalar mean m and variance c
double GaussRand(const double m, const double c){
    
    static unsigned int seed = 0;
    
    boost::mt19937 rng;     
    rng.seed((++seed) + time(NULL));    
    boost::normal_distribution<> NormDist(m, c);
    boost::variate_generator<boost::mt19937&, 
                           boost::normal_distribution<> > generator(rng, NormDist);   
    return(generator());
}

// Generate Gamma R.Vs with shape parameter a and scale parameter b
double GammaRand(double a, double b){
    
    static unsigned int seed = 0;
    
    boost::mt19937 rng1;
    rng1.seed((++seed) + time(NULL));
    boost::gamma_distribution<> GammaDist(a);
    boost::variate_generator<boost::mt19937&, 
                            boost::gamma_distribution<> > generator(rng1, GammaDist);
    
    return(b*generator());
}

// Generate Dirichlet R.Vs with parameters in betaDir
void DirichletRand(double* Pi_Gmm,const double* betaDir, int Nmix){
    
    double sums = 0.0;
            for(unsigned int t=0; t<Nmix; ++t){
                Pi_Gmm[t] = GammaRand(betaDir[t],1);
                sums = sums+Pi_Gmm[t];
            }
            for(unsigned int t=0; t<Nmix; ++t)
                Pi_Gmm[t] = Pi_Gmm[t]/sums;
}

// Generate uniform R.V in the range [0,1)
double UniformGen(void){
    
    static unsigned int seed = 0;
    
    boost::mt19937 rng_u; 
    rng_u.seed((++seed) + time(NULL));
    boost::uniform_real<> uni_dist(0,1);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> > uni(rng_u, uni_dist);
        
        return(uni());
}

//Main
void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
     
    const double* mu0 = (double *)mxGetData(prhs[0]);// Get global mean and set it to mu0   
    const int* dims_X = mxGetDimensions(prhs[0]);// dims_X[0]=sample dimension,dims_X[1]=samplecount
    
    const double* Beta0 = (double *)mxGetData(prhs[1]);// Get global cov and set it to Beta0  
    int Nmix = mxGetScalar(prhs[2]); // Get number of GMM mixtures
    
    int Nstates = mxGetScalar(prhs[3]); // Get number of states
    
    plhs[0] = mxCreateDoubleMatrix(dims_X[0],Nmix,mxREAL); 
    double* mu   = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix(dims_X[0],Nmix,mxREAL); 
    double* cov   = mxGetPr(plhs[1]);
    
    plhs[2] = mxCreateDoubleMatrix(Nmix,1,mxREAL); 
    double* Pi_Gmm   = mxGetPr(plhs[2]);
    
    plhs[3] = mxCreateDoubleMatrix(Nstates,Nstates,mxREAL); 
    double* Aest   = mxGetPr(plhs[3]);
    
 // Initialization of Alpha0,Kappa0

    double* Kappa0 = (double *)mxMalloc(sizeof(double)*dims_X[0]);
    double* Alpha0 = (double *)mxMalloc(sizeof(double)*dims_X[0]);
       
    for(unsigned int i=0;i<dims_X[0];++i){
        Alpha0[i] = 1.0;
        Kappa0[i] = 1.0;
    }
   
    double* Lambda = (double *)mxMalloc(sizeof(double)*dims_X[0]*Nmix);
    
 // Sample Lambda from Alpha0 & Beta0, Sample mu from mu0,Lambda,Kappa0
    for(unsigned int i=0;i<Nmix;++i)
        for(unsigned int j=0;j<dims_X[0];++j){
            Lambda[i*dims_X[0]+j] = GammaRand(Alpha0[j],1/Beta0[j]);
            cov[i*dims_X[0]+j] = 1/Lambda[i*dims_X[0]+j];
            mu[i*dims_X[0]+j] = GaussRand(mu0[j],1/sqrt(Lambda[i*dims_X[0]+j]*Kappa0[j]));
        }
    
 // Sample initial mixture probabilities from Dirichlet Distribution with BetaDir as parameter

    double* BetaDir = (double *)mxMalloc(sizeof(double)*Nmix);
    for(unsigned int i=0;i<Nmix;++i)
        BetaDir[i] = 1.0;          
    DirichletRand(Pi_Gmm,BetaDir,Nmix);
    
 // Sample initial mixture probabilities from Dirichlet Distribution with BetaDir as parameter

    double *BetaDir_Aest = (double *)mxMalloc(sizeof(double)*(Nstates-1));
    
    for(unsigned int i=0;i<Nstates-1;++i)
        BetaDir_Aest[i] = 1.0; 
    
    for(unsigned int i = 0; i<Nstates-1; ++i)
        DirichletRand(&Aest[Nstates*i+i],BetaDir_Aest,Nstates-1);
    
  // Reshape Aest into (Nstates*Nstates) matrix  
    for(unsigned int i = 0; i<Nstates ; ++i)
        for(unsigned int j = 0; j<Nstates ; ++j)
            Aest[Nstates*j+i] = Aest[Nstates*i+j];
    
      for(unsigned int i = 0; i<Nstates ; ++i)
        for(unsigned int j = 0; j<Nstates ; ++j)
            if(j < i || j >= i+(Nstates-1))
                Aest[Nstates*j+i] = 0.0;
    Aest[Nstates*Nstates-1] = 1.0;   // Assign self transition probability for the last state as 1
    
    
    mxFree(Lambda);
    mxFree(Alpha0);
    mxFree(Kappa0);
    mxFree(BetaDir);
    mxFree(BetaDir_Aest);
         
}
