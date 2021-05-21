#include "multiplication.h"
#include "matrix.h"
#include "common.h"
#include "communication.h"

#include <mpi.h>

#include <memory>

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
    communication::Request req;
    std::shared_ptr<PackedData> packedA = std::make_shared<PackedData>(A.pack());

    for (int e = 1; e <= exponent; e++) {
        C = DenseMatrix::blank(B.dim, B.numColumns);

        for (int i = 0; i < rg.size; i++) {
            if (i != rg.size - 1) {
                // Reuse PackedData received from predecessor in replication group,
                // instead of sending SparseMatrix and performing packing, before
                // each send.
                req = communication::Isend<PackedData>(packedA, succ);
            }

            matrixMultiply(A, B, C);

            if (i != rg.size - 1) {
                packedA = std::make_shared<PackedData>(communication::Recv<PackedData>(pred));
                MPI_Wait(req.mpi_request.get(), MPI_STATUS_IGNORE);

                A = SparseMatrix::unpack(*packedA);
            }
        }

        std::swap(B, C);
    }

    return B;
}