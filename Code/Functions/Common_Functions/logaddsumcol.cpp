#include "mex.h"
#include <cmath>
#include <limits>
#include <boost/math/special_functions/log1p.hpp>
  using boost::math::log1p;

// By Timo Korthals
// log add for the colls of a matrix:    [ 1.693147180559945   2.693147180559945   3.693147180559945] = logaddsumcol([1 2 3;1 2 3])

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


double LogAddSum(double* log_matrix, double* log_colSumMatrix, unsigned int NumRows, unsigned int NumCols) {
    
    for(unsigned int IdxCol = 0; IdxCol < NumCols; IdxCol++) {
        log_colSumMatrix[IdxCol] = log_matrix[IdxCol*NumRows];
        for(unsigned int IdxRow = 1; IdxRow < NumRows; IdxRow++) {
            log_colSumMatrix[IdxCol] = LogAdd(log_colSumMatrix[IdxCol], log_matrix[IdxRow+IdxCol*(NumRows)]);
        }
    }
}


void mexFunction(int nlhs,       mxArray *plhs[ ],
                 int nrhs, const mxArray *prhs[ ]) {
    double* log_matrix;
    double* log_colSumMatrix;
    unsigned int NumRows = mxGetM(prhs[0]);
    unsigned int NumCols = mxGetN(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(1, NumCols, mxREAL);
    
    log_matrix = mxGetPr(prhs[0]);
    log_colSumMatrix = mxGetPr(plhs[0]);
    
    LogAddSum(log_matrix, log_colSumMatrix, NumRows, NumCols);
}