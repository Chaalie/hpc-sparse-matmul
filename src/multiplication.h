#ifndef __MULTIPLICATION_H__
#define __MULTIPLICATION_H__

#include "matrix.h"
#include "common.h"
#include "communication.h"

DenseMatrix multiply(Environment& env, SparseMatrix& matA, DenseMatrix& matB, int exponent);

#endif /* __MULTIPLICATION_H__ */