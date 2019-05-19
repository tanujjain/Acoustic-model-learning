
  /*
   * Function to calculate the parameters for a GMM of a given state
   * Usage: [mu,cov,Pi_Gmm] = Sample_GMM(X,Nmix,muPrevious,covPrevious,Pi_GmmPrevious,globmu,globcov);
   * where:
   *        X : Data for a state. (ObsDim*sample_count) matrix
   *        Nmix: Total number of mixtures per GMM
   *        muPrevious : means for each gaussian mixture from previous Iteration. (ObsDim*Nmix) matrix
   *        covPrevious : Covariance for each gaussian mixture from previous Iteration. (ObsDim*Nmix) matrix
   *        Pi_GmmPrevious : weights of each Gaussian in the mixture from previous Iteration. (Nmix*1) vector
   *        mu : means for each gaussian mixture
   *        cov : Diagonal of covariance matrix for each gaussian mixture
   *        Pi_Gmm : weights of each Gaussian in the mixture
   *        globmu :  Global Mean through complete data. (ObsDim*1) vector
   *        globcov :  Global Covariance through complete data. (ObsDim*1) vector
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
double* DirichletRand(double* Pi_Gmm,const double* betaDir, int Nmix){
    
    double sums = 0.0;
            for(unsigned int t=0; t<Nmix; ++t){
                Pi_Gmm[t] = GammaRand(betaDir[t],1);
                sums = sums+Pi_Gmm[t];
            }
            for(unsigned int t=0; t<Nmix; ++t)
                Pi_Gmm[t] = Pi_Gmm[t]/sums;
              
    return Pi_Gmm;
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

// Calculate Gamma for all sample and mixture combinations
void Obs_Prob(double *G,const double *X,const double *mu,const double *cov,
                const double *Pi_Gmm,int ObsDim,int nsamples,int Nmix){        
    double right = 0;
                        
             for(unsigned int i=0;i<Nmix;++i){
                 for(unsigned int j=0;j<nsamples;++j){
                     right = 0;
                     for(unsigned int k=0;k<ObsDim;++k){                                                 
                         right = right + log((cov[ObsDim*i+k])) + (pow(((X[ObsDim*j+k]) - (mu[ObsDim*i+k])),2)/ (cov[ObsDim*i+k])); // log(var(dim,i))+Exponential Part of Gaussian
                     }
                     G[Nmix*j+i] = log(Pi_Gmm[i])-(ObsDim/2.0)*log(2*3.1416)-0.5*right; // log(Pi_Gmm)+(-D/2)*log(2*pi)-(1/2)*right
                 }
             }        
}

// Decide Mixture Label of the sample
int GetLabel(const double* Gamma,double Nmix){
       
    double LogAddVar = Gamma[0];
    for(unsigned int z=1; z<Nmix;++z)     // log(a+b+c+d+e+f+g+h)
        LogAddVar = LogAdd(LogAddVar,Gamma[z]);
    
    double* Gammavec = (double*)mxMalloc(sizeof(double)*Nmix);
    for(unsigned int z=0; z<Nmix;++z)       // Normalizing & converting into linear domain from log domain
        Gammavec[z] = exp(Gamma[z]-LogAddVar);
    
    double* vec = (double*)mxMalloc(sizeof(double)*Nmix);
    vec[0] = Gammavec[0];
    
    for(unsigned int z=1; z<Nmix;++z) // vec = cumsum(Gammavec)
        vec[z]=vec[z-1]+Gammavec[z];
            
        double uni_rv = UniformGen();  // Generate uniform R.V in the range [0,1)
        double SampleLabel = 0;
    for(unsigned int z=0; z<Nmix;++z){
        if(uni_rv < vec[z]){            
            SampleLabel = z;
            break;
        }
        
    }
        
    mxFree(vec);
    mxFree(Gammavec);
    return (SampleLabel);
}

// Main
void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
     
    double* X = (double *)mxGetData(prhs[0]);// Get pointer to elements within sequence    
    const int* dims_X = mxGetDimensions(prhs[0]);// dims_X[0]=sample dimension,dims_X[1]=samplecount
    
    int Nmix = mxGetScalar(prhs[1]); // Get number of GMM mixtures
    
    const double* muPrev = mxGetPr(prhs[2]);
    const double* covPrev = mxGetPr(prhs[3]);
    const double* Pi_GmmPrev = mxGetPr(prhs[4]);
    
    const double* mu0 = mxGetPr(prhs[5]); // Gather global mean across all data and set it to mu0
    const double* Beta0 = mxGetPr(prhs[6]); // Gather global cov across all data and set it to beta0
    
    
    plhs[0] = mxCreateDoubleMatrix(dims_X[0],Nmix,mxREAL); 
    double* mu   = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix(dims_X[0],Nmix,mxREAL); 
    double* cov   = mxGetPr(plhs[1]);
    
    plhs[2] = mxCreateDoubleMatrix(1,Nmix,mxREAL); 
    double* Pi_Gmm   = mxGetPr(plhs[2]);
    
 // Initialization of Alpha0,Kappa0
    
    double* Alpha0 = (double *)mxMalloc(sizeof(double)*dims_X[0]);
    double* AlphaUpdated = (double *)mxMalloc(sizeof(double)*dims_X[0]*Nmix);
    double* BetaGamUpdated = (double *)mxMalloc(sizeof(double)*dims_X[0]*Nmix);
    double* Kappa0 = (double *)mxMalloc(sizeof(double)*dims_X[0]);
    double* KappaUpdated = (double *)mxMalloc(sizeof(double)*dims_X[0]*Nmix);
    
    for(unsigned int i=0;i<dims_X[0];++i){
        Alpha0[i] = 1.0;
        Kappa0[i] = 1.0;
    }
    
    double *Lambda = (double *)mxMalloc(sizeof(double)*dims_X[0]*Nmix);
    double* Mean_sample = (double *)mxMalloc(sizeof(double)*dims_X[0]);
    
    double *BetaDir = (double *)mxMalloc(sizeof(double)*Nmix);
    for(unsigned int i=0;i<Nmix;++i)
        BetaDir[i] = 1.0;          

    double *BetaDirUpdated = (double *)mxMalloc(sizeof(double)*Nmix); // Allocate space for updated dirichlet parameter
    double *Gamma = (double *)mxMalloc(sizeof(double)*dims_X[0]*Nmix*dims_X[1]);
    int *Labels = (int *)mxMalloc(sizeof(int)*dims_X[1]);
    
 // Iterations begin
    unsigned int NumIter =1;
    
    for(unsigned int it = 0; it<NumIter ; ++it){
        
        Obs_Prob(Gamma,X,muPrev,covPrev,Pi_GmmPrev,dims_X[0],dims_X[1],Nmix); // Gamma : sample1[mix-1 ..mix-8],sample2[mix-1 ..mix-8]        
        int count[Nmix];
        
        for(unsigned int i=0;i<Nmix;++i)  // Initialize number of samples per mixture to 0
        count[i] = 0;        
        
        for(unsigned int n=0; n<dims_X[1] ; ++n){  // Get mixture Label to each sample based on Gamma
            Labels[n] = GetLabel(&Gamma[Nmix*n],Nmix);
            count[Labels[n]] = count[Labels[n]]+1;
        }
 
  // Updates begin here
                       
        for(unsigned int i=0;i<Nmix;++i){  // Goes over each mixture
            BetaDirUpdated[i] = BetaDir[i]+count[i];  
            for(unsigned int dim=0;dim<dims_X[0];++dim){
                AlphaUpdated[i*dims_X[0]+dim] = Alpha0[dim]+0.5*count[i];
                KappaUpdated[i*dims_X[0]+dim] = Kappa0[dim]+count[i];
            }
            
             double classmean[dims_X[0]];
             double classvar[dims_X[0]];
             
             for(unsigned int l=0;l<dims_X[0];++l)
                 classmean[l] = 0.0;        
                   
             for(unsigned int l=0;l<dims_X[0];++l)
                 classvar[l] = 0.0; 
             
             for(unsigned int j=0;j<dims_X[1];++j){
                if(Labels[j] == i){                       
                    for(unsigned int l=0;l<dims_X[0];++l)
                        classmean[l] = classmean[l]+X[dims_X[0]*j+l];                                                   
                }                                        
              }
             
             if(count[i] != 0){
                for(unsigned int l=0;l<dims_X[0];++l){
                    classmean[l] = classmean[l]/count[i];
                }
             }
             // Covariance
             for(unsigned int j=0;j<dims_X[1];++j){
                 if(Labels[j] == i){                        
                     for(unsigned int l=0;l<dims_X[0];++l)
                        classvar[l] = classvar[l] + pow((X[j*dims_X[0]+l] - classmean[l]),2);                                                   
                 }                                        
              }
            
             // BetaGamUpdated, Lambda,mu Updated
             
             for(unsigned int k=0; k<dims_X[0]; ++k){
                 BetaGamUpdated[i*dims_X[0]+k] = Beta0[k]+0.5*classvar[k]+
                         (0.5*Kappa0[k]*count[i]*pow((classmean[k]-mu0[k]),2)/KappaUpdated[i*dims_X[0]+k]);
                 
                 Lambda[i*dims_X[0]+k] = GammaRand(AlphaUpdated[i*dims_X[0]+k],1/BetaGamUpdated[i*dims_X[0]+k]);
                 cov[i*dims_X[0]+k] = 1/Lambda[i*dims_X[0]+k];
                 Mean_sample[k] = (Kappa0[k]*mu0[k]+count[i]*classmean[k])/KappaUpdated[i*dims_X[0]+k];
                 mu[i*dims_X[0]+k] = GaussRand(Mean_sample[k],1/sqrt(Lambda[i*dims_X[0]+k]*KappaUpdated[i*dims_X[0]+k]));
             }
             
        }
        
        Pi_Gmm = DirichletRand(Pi_Gmm,BetaDirUpdated,Nmix); // Sample Pi_Gmm from Dirichlet Distribution with updated parameter
        
   // Updates end here    
        // Iterations end
    }    
        
    // Free memory
    
    mxFree(Gamma);
    mxFree(Labels);
    mxFree(Kappa0);
    mxFree(Alpha0);
    mxFree(Lambda);
    mxFree(Mean_sample);
    mxFree(AlphaUpdated);
    mxFree(BetaGamUpdated);
    mxFree(KappaUpdated);
    mxFree(BetaDirUpdated);
    mxFree(BetaDir);
    
    
}