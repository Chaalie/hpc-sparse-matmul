#include <mpi.h>

#include <memory>

#include "common.h"
#include "context.h"
#include "matrix.h"
#include "multiplication.h"

// Perform C += A * B, C does not have to be blank (zeroes)
void matrixMultiply(SparseMatrix& A, DenseMatrix& B, DenseMatrix& C) {
    for (int c = 0; c < B.dimension.col; c++) {
        for (auto fieldA : A) {
            MatrixIndex idxA;
            double valueA;
            std::tie(idxA, valueA) = fieldA;

            C(idxA.row, c) += valueA * B(idxA.col, c);
        }
    }
}

DenseMatrix multiply(Context& ctx, SparseMatrix&& inA, DenseMatrix&& inB, int exponent) {
    SparseMatrix matA = std::move(inA);
    DenseMatrix matB = std::move(inB);
    DenseMatrix matC;
    MPI_Request sendReq[2], recvReq;
    PackedData sendData, recvData;
    int sendSize, recvSize;
    int numShifts = ctx.numReplicationGroups / ctx.replicationGroupSize;
    int isRGLeader = ctx.process.sparseRG.isLeader(ctx.process.id);

    if (isRGLeader) {
        sendData = pack<SparseMatrix>(matA, ctx.process.sparseRG.predInterComm);
    }

    for (int e = 1; e <= exponent; e++) {
        matC = DenseMatrix::blank(matB.dimension);

        for (int i = 1; i <= numShifts; i++) {
            if (i != numShifts) {
                if (isRGLeader) {
                    // Reuse PackedData received from predecessor in replication group,
                    // instead of sending SparseMatrix and performing packing, before
                    // each send.
                    sendSize = sendData.size();
                    MPI_Ibcast(&sendSize, 1, MPI_INT, MPI_ROOT, ctx.process.sparseRG.predInterComm, &sendReq[0]);
                    MPI_Ibcast(sendData.data(), sendData.size(), MPI_PACKED, MPI_ROOT,
                               ctx.process.sparseRG.predInterComm, &sendReq[1]);
                }
                MPI_Bcast(&recvSize, 1, MPI_INT, INTERNAL_LEADER_ID, ctx.process.sparseRG.succInterComm);
                MPI_Ibcast(recvData.data(), recvData.size(), MPI_PACKED, INTERNAL_LEADER_ID,
                           ctx.process.sparseRG.succInterComm, &recvReq);
            }

            matrixMultiply(matA, matB, matC);

            if (i != numShifts) {
                MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
                if (isRGLeader) {
                    MPI_Waitall(2, sendReq, MPI_STATUSES_IGNORE);
                }

                matA = unpack<SparseMatrix>(recvData, ctx.process.sparseRG.succInterComm);

                if (isRGLeader) {
                    sendData = std::move(recvData);
                }
            }
        }

        // if (ctx.process.denseRG.size > 1) {
        //     MPI_Allreduce(MPI_IN_PLACE, B.data, B.size, MPI_DenseColumn, MPI_AddDenseColumn,
        //     MPI_DenseReplicationComm);
        // }

        std::swap(matB, matC);
    }

    return matB;
}