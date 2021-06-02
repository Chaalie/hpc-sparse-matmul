#include <mpi.h>

#include <memory>

#include "common.h"
#include "context.h"
#include "matrix.h"
#include "multiplication.h"
#include "mpi_helpers.h"

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
    MPI_Request sendReq, recvReq;
    PackedData sendData, recvData;
    int sendSize;
    std::vector<int> recvSizeCache(ctx.numReplicationGroups, -1);
    int matFragIdx = 0;

    int numShifts;
    switch (ctx.algorithm) {
        case Algorithm::ColumnA:
            numShifts = ctx.numReplicationGroups;
            break;
        case Algorithm::InnerABC:
            numShifts = ctx.numReplicationGroups / ctx.replicationGroupSize;
            break;
        default:
            throw "should not happen";
    }

    int isRGLeader = ctx.process.sparseRG.isLeader(ctx.process.id);
    if (isRGLeader) {
        sendData = pack<SparseMatrix>(matA, ctx.process.sparseRG.predInterComm);
    }

    for (int e = 1; e <= exponent; e++) {
        matC = DenseMatrix::blank(matB.dimension);

        for (int i = 1; i <= numShifts; i++) {
            // if (ctx.process.id == 0) {
            //     matA.print(1);
            //     std::cout << "====" << std::endl;
            // }
            if (i != numShifts) {
                // Processes are unaware about size of packed data they will receive, thus it need
                // to be sent (broadcasted) to them.
                // But over time matrix fragments received by the processes will duplicate, as processes
                // loops through entire sparse matrix in the span of the multiplication. Thus, we can
                // cache those expected receive sizes.
                if (recvSizeCache[matFragIdx] == -1) {
                    if (isRGLeader) {
                        sendSize = sendData.size();
                        MPI_Ibcast(&sendSize, 1, MPI_INT, MPI_ROOT, ctx.process.sparseRG.predInterComm, &sendReq);
                    }
                    MPI_Ibcast(&recvSizeCache[matFragIdx], 1, MPI_INT, INTERNAL_LEADER_ID,
                               ctx.process.sparseRG.succInterComm, &recvReq);

                    MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
                    if (isRGLeader) {
                        MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
                    }
                }

                if (isRGLeader) {
                    MPI_Ibcast(sendData.data(), sendData.size(), MPI_PACKED, MPI_ROOT,
                               ctx.process.sparseRG.predInterComm, &sendReq);
                }
                recvData.resize(recvSizeCache[matFragIdx]);
                MPI_Ibcast(recvData.data(), recvData.size(), MPI_PACKED, INTERNAL_LEADER_ID,
                           ctx.process.sparseRG.succInterComm, &recvReq);
                matFragIdx = (matFragIdx + 1) % ctx.numReplicationGroups;
            }

            matrixMultiply(matA, matB, matC);

            if (i != numShifts) {
                MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
                if (isRGLeader) {
                    MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
                }

                matA = unpack<SparseMatrix>(recvData, ctx.process.sparseRG.succInterComm);

                if (isRGLeader) {
                    // Reuse PackedData received from predecessor in replication group,
                    // instead of sending SparseMatrix and performing packing, before
                    // each send.
                    sendData = std::move(recvData);
                }
            }
        }

        if (ctx.process.denseRG.size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, matC.data.data(), matC.data.size(), MPI_DOUBLE, MPI_SUM,
                          ctx.process.denseRG.internalComm);
        }

        std::swap(matB, matC);
    }

    return matB;
}