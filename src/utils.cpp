#include "common.h"
#include "context.h"
#include "matrix.h"
#include "utils.h"

int utils::initializeMatrixDimension(int processId, SparseMatrix& matrix) {
    int dimension;
    if (isMainLeader(processId)) {
        dimension = matrix.dimension.col;
    }
    MPI_Bcast(&dimension, 1, MPI_INT, MAIN_LEADER_ID, MPI_COMM_WORLD);

    return dimension;
}

SparseMatrix utils::initializeSparseMatrix(Context& ctx, SparseMatrix& wholeMatrix) {
    SparseMatrixReplicationGroup rg = ctx.process.sparseRG;
    int recvSize;                                          // size of data scattered to process
    PackedData recvData;                                   // data scattered to process
    PackedData accSendData;                                // accumulated packed data used for scatter
    std::vector<int> sendSizes(ctx.numProcesses);          // size of each process'es data
    std::vector<int> sendDisplacements(ctx.numProcesses);  // displacement of each process'es data

    if (ctx.process.isMainLeader()) {
        // Send to each process its fragment of the matrix
        // distribute sparse matrix
        for (int p = 0; p < ctx.numProcesses; p++) {
            MatrixFragment frag = utils::getProcessSparseFragment(ctx, p);
            std::cout << "Fragment of " << p << " " << std::get<0>(frag) << " " << std::get<1>(frag) << std::endl;
            auto matrixFragment = std::move(wholeMatrix.maskSubMatrix(frag));
            matrixFragment.print(1);
            auto packedMatrixFragment = pack<SparseMatrix>(matrixFragment, MPI_COMM_WORLD);

            sendSizes[p] = packedMatrixFragment.size();
            sendDisplacements[p] = accSendData.size();
            accSendData.insert(accSendData.end(), std::make_move_iterator(packedMatrixFragment.begin()),
                               std::make_move_iterator(packedMatrixFragment.end()));
        }
    }
    // send information of packed data size to receive by the process
    MPI_Scatter(sendSizes.data(), 1, MPI_INT, &recvSize, 1, MPI_INT, MAIN_LEADER_ID, MPI_COMM_WORLD);
    recvData.resize(recvSize);

    // distribute initial sparse matrix fragments across processes
    MPI_Scatterv(accSendData.data(), sendSizes.data(), sendDisplacements.data(), MPI_PACKED, recvData.data(), recvSize,
                 MPI_PACKED, MAIN_LEADER_ID, MPI_COMM_WORLD);

    int rgAccRecvSize = 0;                                             // total size of packed data in replication group
    std::vector<int> rgPackedSizes(rg.size);          // size of each member's packed data
    std::vector<int> rgPackedDisplacements(rg.size);  // displacement of each member's packed data
    PackedData rgAccRecvData;                                          // accumulated packed data of replication group

    // gather information about size of data held by each replication group member
    MPI_Allgather(&recvSize, 1, MPI_INT, rgPackedSizes.data(), 1, MPI_INT, rg.internalComm);

    for (int i = 0; i < (int)rgPackedSizes.size(); i++) {
        rgPackedDisplacements[i] = rgAccRecvSize;
        rgAccRecvSize += rgPackedSizes[i];
    }
    rgAccRecvData.resize(rgAccRecvSize);

    // gather packed data within replication group
    MPI_Allgatherv(recvData.data(), recvData.size(), MPI_PACKED, rgAccRecvData.data(), rgPackedSizes.data(),
                   rgPackedDisplacements.data(), MPI_PACKED, rg.internalComm);

    // unpack and reconstruct replication group's matrix fragment
    SparseMatrix resultMatrix = std::move(SparseMatrix::blank({ctx.matrixDimension, ctx.matrixDimension}));
    for (int i = 0; i < rg.size; i++) {
        auto matFrag = unpack<SparseMatrix>(rgAccRecvData.data() + rgPackedDisplacements[i], rgPackedSizes[i], rg.internalComm);
        resultMatrix.join(std::move(matFrag));
    }

    return resultMatrix;
}

/*
    Performs a fair chunk split of size @size into @numParts parts. 
    Split is made so that each difference between parts is as small as possible (max 1).
    Returns a beginning of part @partId.
*/
int getFairPartBeginning(int partId, int size, int numParts) {
    int basePartSize = size / numParts;
    int numBiggerParts = std::min(partId, size % numParts);

    return partId * basePartSize + numBiggerParts;
}

std::tuple<int, int> getProcessSparseCoordinates(Context& ctx, int processId) {
    int rgId;
    int idWithinRg;
    switch (ctx.algorithm) {
        case Algorithm::ColumnA:
            rgId = processId / ctx.replicationGroupSize;
            idWithinRg = processId % ctx.replicationGroupSize;
            break;
        case Algorithm::InnerABC:
            int numShifts = ctx.numReplicationGroups / ctx.replicationGroupSize;  // shifts required for a single multiplication
            int denseLayerId = processId % ctx.numReplicationLayers;
            int denseGroupId = processId / ctx.replicationGroupSize;
            rgId = (denseLayerId * numShifts) + (denseGroupId % numShifts);
            idWithinRg = processId / ctx.numReplicationGroups;
            break;
    }

    return {rgId, idWithinRg};
}

MatrixFragment utils::getProcessSparseFragment(Context& ctx, int processId) {
    std::cout << "Calc sparse fragment for " << processId << std::endl;
    int rgId;
    int idWithinRg;
    std::tie(rgId, idWithinRg) = getProcessSparseCoordinates(ctx, processId);
    std::cout << "Got coordinates " << processId << " " << rgId << " " << idWithinRg << std::endl;

    int rgFragmentStart = getFairPartBeginning(rgId, ctx.matrixDimension, ctx.numReplicationGroups);
    int rgFragmentEnd = getFairPartBeginning(rgId + 1, ctx.matrixDimension, ctx.numReplicationGroups);
    int rgFragmentSize = rgFragmentEnd - rgFragmentStart;
    std::cout << "rgFragment got " << rgFragmentStart << " " << rgFragmentEnd << std::endl;

    int processFragmentStart = rgFragmentStart + getFairPartBeginning(idWithinRg, rgFragmentSize, ctx.replicationGroupSize);
    int processFragmentEnd = rgFragmentStart + getFairPartBeginning(idWithinRg + 1, rgFragmentSize, ctx.replicationGroupSize) - 1;

    int rowStart, rowEnd;
    int columnStart, columnEnd;
    switch (ctx.algorithm) {
        case Algorithm::ColumnA:
            rowStart = 0;
            rowEnd = ctx.matrixDimension;
            columnStart = processFragmentStart;
            columnEnd = processFragmentEnd;
            break;
        case Algorithm::InnerABC:
            columnStart = 0;
            columnEnd = ctx.matrixDimension;
            rowStart = processFragmentStart;
            rowEnd = processFragmentEnd;
            break;
    }

    return {{rowStart, columnStart}, {rowEnd, columnEnd}};
}

std::tuple<int, int> getProcessDenseCoordinates(Context& ctx, int processId) {
    return {0, 0};
}

MatrixFragment utils::getProcessDenseFragment(Context& ctx, int processId) {
    return {{0, 0}, {0, 0}};
}

DenseMatrix utils::initializeDenseMatrix(Context& ctx, int denseMatrixSeed) { return DenseMatrix(); }