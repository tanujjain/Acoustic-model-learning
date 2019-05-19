
/*
   * Function to implement log addition
   * Usage: LogSumAB = logaddsum(vector);
   * where:
   *        vector : A number of elements times 1 vector which needs to be added
   *        LogSumAB : the returned sum
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


double LogAddSum(double* ab, unsigned int NumInputElems) {
    double LogSumAB = ab[0];
    
    for(unsigned int IdxElem = 1; IdxElem < NumInputElems; IdxElem++) {
        LogSumAB = LogAdd(LogSumAB, ab[IdxElem]);
    }
    
    return(LogSumAB);
}


void mexFunction(int nlhs,       mxArray *plhs[ ],
                 int nrhs, const mxArray *prhs[ ]) {
    double* ab;
    double* LogSumAB;
    unsigned int NumInputElems = mxGetM(prhs[0])*mxGetN(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    ab = mxGetPr(prhs[0]);
    LogSumAB = mxGetPr(plhs[0]);
    
    LogSumAB[0] = LogAddSum(ab, NumInputElems);
}