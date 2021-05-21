#include "multiplication.h"
#include "matrix.h"
#include "common.h"
#include "communication.h"

#include <mpi.h>

// Perform C += A * B, C does not have to be blank (zeroes)
void matrixMultiply(SparseMatrix& A, DenseMatrix& B, DenseMatrix& C) {
    for (int c = 0; c < B.numColumns; c++) {
        for (auto fieldA: A) {
            MatrixIndex idxA;
            double valueA;
            std::tie(idxA, valueA) = fieldA;

            C.values[idxA.row][c] += valueA * B.values[idxA.col][c];
        }
    }
}

DenseMatrix multiply(Environment& env, SparseMatrix& baseA, DenseMatrix& baseB, int exponent) {

    SparseMatrix A = std::move(baseA);
    DenseMatrix B = std::move(baseB);
    DenseMatrix C;
    ReplicationGroup rg = ReplicationGroup::ofProcess(env, env.localId);
    int succ = rg.successorOf(env.localId);
    int pred = rg.predecessorOf(env.localId);
    MPI_Request request;
    PackedData packedA = A.pack();

    for (int e = 1; e <= exponent; e++) {
        C = DenseMatrix::blank(B.dim, B.numColumns);

        for (int i = 0; i < rg.size; i++) {
            if (i != rg.size - 1) {
                // Reuse PackedData received from predecessor in replication group,
                // instead of sending SparseMatrix and performing packing, before
                // each send.
                communication::Isend<PackedData>(packedA, succ, request);
            }

            matrixMultiply(A, B, C);

            if (i != rg.size - 1) {
                packedA = communication::Recv<PackedData>(pred);
                MPI_Wait(&request, MPI_STATUS_IGNORE);

                A = SparseMatrix::unpack(packedA);
            }
        }

        std::swap(B, C);
    }

    return B;
}