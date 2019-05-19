#include "mex.h"
#include <cmath>
#include <limits>
#include <boost/math/special_functions/log1p.hpp>
  using boost::math::log1p;

// By Timo Korthals
// log add for the two vector:    [0 4.1 9]' = logaddsum2([1 2 3]',[1 4 9]')


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


void LogAddSum(double* a, double* b, double* LogSumAB, unsigned int NumInputElems) {
    
    for(unsigned int IdxElem = 0; IdxElem < NumInputElems; IdxElem++) {
        LogSumAB[IdxElem] = LogAdd(a[IdxElem], b[IdxElem]);
    }
}


void mexFunction(int nlhs,       mxArray *plhs[ ],
                 int nrhs, const mxArray *prhs[ ]) {
    double* a;
    double* b;
    double* LogSumAB;
    unsigned int NumInputElems = mxGetM(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(NumInputElems, 1, mxREAL);
    
    a = mxGetPr(prhs[0]);
    b = mxGetPr(prhs[1]);
    LogSumAB = mxGetPr(plhs[0]);
    
    LogAddSum(a ,b , LogSumAB, NumInputElems);
}