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
            auto matrixFragment = std::move(wholeMatrix.maskSubMatrix(frag));
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

    int rgAccRecvSize = 0;                            // total size of packed data in replication group
    std::vector<int> rgPackedSizes(rg.size);          // size of each member's packed data
    std::vector<int> rgPackedDisplacements(rg.size);  // displacement of each member's packed data
    PackedData rgAccRecvData;                         // accumulated packed data of replication group

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
        auto matFrag =
            unpack<SparseMatrix>(rgAccRecvData.data() + rgPackedDisplacements[i], rgPackedSizes[i], rg.internalComm);
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

std::tuple<int, int> getRgMemberFragment(int rgId, int idWithinRg, int matrixDimension, int numReplicationGroups,
                                         int replicationGroupSize) {
    int rgFragmentStart = getFairPartBeginning(rgId, matrixDimension, numReplicationGroups);
    int rgFragmentEnd = getFairPartBeginning(rgId + 1, matrixDimension, numReplicationGroups);
    int rgFragmentSize = rgFragmentEnd - rgFragmentStart;

    int fragmentStart = rgFragmentStart + getFairPartBeginning(idWithinRg, rgFragmentSize, replicationGroupSize);
    int fragmentEnd = rgFragmentStart + getFairPartBeginning(idWithinRg + 1, rgFragmentSize, replicationGroupSize);

    return {fragmentStart, fragmentEnd};
}

std::tuple<int, int> getProcessSparseCoordinates(Context& ctx, int processId) {
    int rgId;
    int idWithinRg;
    switch (ctx.algorithm) {
        case Algorithm::ColumnA:
            rgId = processId / ctx.replicationGroupSize;
            idWithinRg = processId % ctx.replicationGroupSize;
            break;
        case Algorithm::InnerABC: {
            int numShifts =
                ctx.numReplicationGroups / ctx.replicationGroupSize;  // shifts required for a single multiplication
            int denseLayerId = processId % ctx.numReplicationLayers;
            int denseGroupId = processId / ctx.replicationGroupSize;
            rgId = (denseLayerId * numShifts) + (denseGroupId % numShifts);
            idWithinRg = processId / ctx.numReplicationGroups;
            break;
        }
        default:
            throw "should not happen";
    }

    return {rgId, idWithinRg};
}

MatrixFragment utils::getProcessSparseFragment(Context& ctx, int processId) {
    int rgId, idWithinRg;
    std::tie(rgId, idWithinRg) = getProcessSparseCoordinates(ctx, processId);

    int processFragmentStart, processFragmentEnd;
    std::tie(processFragmentStart, processFragmentEnd) =
        getRgMemberFragment(rgId, idWithinRg, ctx.matrixDimension, ctx.numReplicationGroups, ctx.replicationGroupSize);

    switch (ctx.algorithm) {
        case Algorithm::ColumnA:
            return {{0, processFragmentStart}, {ctx.matrixDimension, processFragmentEnd}};
        case Algorithm::InnerABC:
            return {{processFragmentStart, 0}, {processFragmentEnd, ctx.matrixDimension}};
        default:
            throw "should not happen";
    }
}

std::tuple<int, int> getProcessDenseCoordinates(Context& ctx, int processId) {
    int rgId;
    int idWithinRg;
    switch (ctx.algorithm) {
        case Algorithm::ColumnA:
            rgId = processId;
            idWithinRg = 0;
            break;
        case Algorithm::InnerABC:
            rgId = processId / ctx.replicationGroupSize;
            idWithinRg = processId % ctx.replicationGroupSize;
            break;
        default:
            throw "should not happen";
    }
    return {rgId, idWithinRg};
}

MatrixFragment utils::getProcessDenseFragment(Context& ctx, int processId) {
    int rgId, idWithinRg;
    std::tie(rgId, idWithinRg) = getProcessDenseCoordinates(ctx, processId);

    int numReplicationGroups, replicationGroupSize;
    switch (ctx.algorithm) {
        case Algorithm::ColumnA:
            numReplicationGroups = ctx.numProcesses;
            replicationGroupSize = 1;
            break;
        case Algorithm::InnerABC:
            numReplicationGroups = ctx.numReplicationGroups;
            replicationGroupSize = ctx.replicationGroupSize;
            break;
        default:
            throw "should not happen";
    }

    int processFragmentStart, processFragmentEnd;
    std::tie(processFragmentStart, processFragmentEnd) =
        getRgMemberFragment(rgId, idWithinRg, ctx.matrixDimension, numReplicationGroups, replicationGroupSize);

    return {{0, processFragmentStart}, {ctx.matrixDimension, processFragmentEnd}};
}

DenseMatrix utils::initializeDenseMatrix(Context& ctx, int denseMatrixSeed) {
    ReplicationGroup rg = ctx.process.denseRG;
    auto frag = getProcessDenseFragment(ctx, ctx.process.id);
    auto matrixFragment = DenseMatrix::generate(frag, denseMatrixSeed);
    auto packedMatrixFragment = pack<DenseMatrix>(matrixFragment, rg.internalComm);

    int rgAccRecvSize = 0;                            // total size of packed data in replication group
    std::vector<int> rgPackedSizes(rg.size);          // size of each member's packed data
    std::vector<int> rgPackedDisplacements(rg.size);  // displacement of each member's packed data
    PackedData rgAccRecvData;                         // accumulated packed data of replication group

    // gather information about size of data held by each replication group member
    int packedSize = packedMatrixFragment.size();
    MPI_Allgather(&packedSize, 1, MPI_INT, rgPackedSizes.data(), 1, MPI_INT, rg.internalComm);

    for (int i = 0; i < (int)rgPackedSizes.size(); i++) {
        rgPackedDisplacements[i] = rgAccRecvSize;
        rgAccRecvSize += rgPackedSizes[i];
    }
    rgAccRecvData.resize(rgAccRecvSize);

    // gather packed data within replication group
    MPI_Allgatherv(packedMatrixFragment.data(), packedMatrixFragment.size(), MPI_PACKED, rgAccRecvData.data(),
                   rgPackedSizes.data(), rgPackedDisplacements.data(), MPI_PACKED, rg.internalComm);

    // unpack and reconstruct replication group's matrix fragment
    DenseMatrix resultMatrix = std::move(DenseMatrix::blank({ctx.matrixDimension, 0}));
    for (int i = 0; i < rg.size; i++) {
        auto matFrag =
            unpack<DenseMatrix>(rgAccRecvData.data() + rgPackedDisplacements[i], rgPackedSizes[i], rg.internalComm);
        resultMatrix.join(std::move(matFrag));
    }

    return resultMatrix;
}

DenseMatrix utils::gatherDenseMatrix(Context& ctx, DenseMatrix& matrix, int gatherTo) {
    DenseMatrixReplicationGroup rg = ctx.process.denseRG;
    DenseMatrix result = DenseMatrix::blank({matrix.dimension.row, matrix.dimension.row});

    if (rg.isLeader(ctx.process.id)) {
        int leadersCount;
        MPI_Comm_size(rg.leadersComm, &leadersCount);
        std::vector<int> recvSizes(leadersCount);
        std::vector<int> recvDisplacements(leadersCount);

        int matrixSize = matrix.data.size();
        MPI_Gather(&matrixSize, 1, MPI_INT, recvSizes.data(), 1, MPI_INT, gatherTo, rg.leadersComm);

        if (ctx.process.id == gatherTo) {
            for (int i = 1; i < leadersCount; i++) {
                recvDisplacements[i] = recvDisplacements[i - 1] + recvSizes[i - 1];
            }
        }
        MPI_Gatherv(matrix.data.data(), matrix.data.size(), MPI_DOUBLE, result.data.data(), recvSizes.data(),
                    recvDisplacements.data(), MPI_DOUBLE, gatherTo, rg.leadersComm);
    }
    return result;
}

int utils::gatherCountGE(Context& ctx, DenseMatrix& matrix, int geValue, int gatherTo) {
    int numReplicationGroups = ctx.algorithm == Algorithm::ColumnA ? ctx.numProcesses : ctx.numReplicationGroups;
    DenseMatrixReplicationGroup rg = ctx.process.denseRG;

    int rgFragmentStart = getFairPartBeginning(rg.id, ctx.matrixDimension, numReplicationGroups);
    MatrixIndex processFragmentStart, processFragmentEnd;
    std::tie(processFragmentStart, processFragmentEnd) = utils::getProcessDenseFragment(ctx, ctx.process.id);
    processFragmentStart.col -= rgFragmentStart;
    processFragmentEnd.col -= rgFragmentStart;
    MatrixFragment processFragment = {processFragmentStart, processFragmentEnd};

    int geCount = matrix.countGE(processFragment, geValue);
    int geCountRet = -1;

    MPI_Reduce(&geCount, &geCountRet, 1, MPI_INT, MPI_SUM, INTERNAL_LEADER_ID, rg.internalComm);
    if (rg.isLeader(ctx.process.id)) {
        geCount = geCountRet;
        MPI_Reduce(&geCount, &geCountRet, 1, MPI_INT, MPI_SUM, gatherTo, rg.leadersComm);
    }

    return geCountRet;
}