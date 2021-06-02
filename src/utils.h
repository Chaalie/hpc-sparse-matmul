#ifndef __UTILS_H__
#define __UTILS_H__

#include <mpi.h>

#include <string>

#include "common.h"
#include "context.h"
#include "matrix.h"

namespace utils {

int initializeMatrixDimension(int processId, SparseMatrix& matrix);

SparseMatrix initializeSparseMatrix(Context& ctx, SparseMatrix& matrix);

DenseMatrix initializeDenseMatrix(Context& ctx, int denseMatrixSeed);

MatrixFragment getProcessDenseFragment(Context& ctx, int processId);

MatrixFragment getProcessSparseFragment(Context& ctx, int processId);

DenseMatrix gatherDenseMatrix(Context& ctx, DenseMatrix& matrix, int gatherTo);

int gatherCountGE(Context& ctx, DenseMatrix& matrix, double geValue, int gatherTo);

void verifyPreconditions(int p, int c, Algorithm algorithm);
};  // namespace utils

#endif /* __UTILS_H__ */