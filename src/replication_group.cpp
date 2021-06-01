#include <mpi.h>

#include "common.h"
#include "replication_group.h"

/*
    In ColumnA we do not replicate dense matrix so each replication group consists
    of a single process, thus there's a single replication layer.
*/
template <>
DenseMatrixReplicationGroup DenseMatrixReplicationGroup::ofProcess<Algorithm::ColumnA>(int processId, int numProcesses,
                                                                                       int, int) {
    std::cout << "START: dense matrix ColumnA " << processId << " " << numProcesses << std::endl;
    int replicationGroupSize = 1;
    MPI_Comm localComm = MPI_COMM_SELF;

    std::cout << "END: dense matrix ColumnA " << processId << " " << numProcesses << std::endl;
    return DenseMatrixReplicationGroup(processId, replicationGroupSize, processId, localComm);
}

/*
    Replication groups are constructed from consecutive processes e.g.
    P = 9
    C = 3
    Replication groups:
        0 - [0, 1, 2]
        1 - [3, 4, 5]
        2 - [6, 7, 8]
*/
template <>
SparseMatrixReplicationGroup SparseMatrixReplicationGroup::ofProcess<Algorithm::ColumnA>(int processId,
                                                                                         int numProcesses,
                                                                                         int numReplicationGroups,
                                                                                         int replicationGroupSize) {
    int localRgId = processId / replicationGroupSize;
    int localRgGlobalId = SparseMatrixReplicationGroup::getGlobalId(localRgId, numReplicationGroups);
    int localLeaderId = localRgId * replicationGroupSize;

    MPI_Comm localComm;
    MPI_Comm_split(MPI_COMM_WORLD, localRgGlobalId, processId, &localComm);

    int predRgId = (localRgId - 1 + numReplicationGroups) % numReplicationGroups;
    int predRgGlobalId = SparseMatrixReplicationGroup::getGlobalId(predRgId, numReplicationGroups);
    int predRgLeaderId = predRgId * replicationGroupSize;
    MPI_Comm predInterComm;

    int succRgId = (localRgId + 1) % numReplicationGroups;
    int succRgGlobalId = SparseMatrixReplicationGroup::getGlobalId(succRgId, numReplicationGroups);
    int succRgLeaderId = succRgId * replicationGroupSize;
    MPI_Comm succInterComm;

    if (localRgId % 2 == 0) {
        MPI_Intercomm_create(localComm, INTERNAL_LEADER_ID, MPI_COMM_WORLD, succRgLeaderId, localRgGlobalId,
                             &succInterComm);
        if (processId == localLeaderId) {
            MPI_Intercomm_create(MPI_COMM_SELF, INTERNAL_LEADER_ID, MPI_COMM_WORLD, predRgLeaderId, predRgGlobalId,
                                 &predInterComm);
        }
    } else {
        if (processId == localLeaderId) {
            MPI_Intercomm_create(MPI_COMM_SELF, INTERNAL_LEADER_ID, MPI_COMM_WORLD, predRgLeaderId, predRgGlobalId,
                                 &predInterComm);
        }
        MPI_Intercomm_create(localComm, INTERNAL_LEADER_ID, MPI_COMM_WORLD, succRgLeaderId, localRgGlobalId,
                             &succInterComm);
    }

    return SparseMatrixReplicationGroup(localRgId, replicationGroupSize, localLeaderId, localComm, predInterComm,
                                        succInterComm);
}

/*
    Same groups assignment as for ColumnA sparse matrix is used.
*/
template <>
DenseMatrixReplicationGroup DenseMatrixReplicationGroup::ofProcess<Algorithm::InnerABC>(int processId, int numProcesses,
                                                                                        int numReplicationGroups,
                                                                                        int replicationGroupSize) {
    int localRgId = processId % numReplicationGroups;
    int localRgGlobalId = DenseMatrixReplicationGroup::getGlobalId(localRgId, numReplicationGroups);
    int localLeaderId = localRgId * replicationGroupSize;
    MPI_Comm localComm;
    MPI_Comm_split(MPI_COMM_WORLD, localRgGlobalId, processId, &localComm);

    return DenseMatrixReplicationGroup(localRgId, replicationGroupSize, localLeaderId, localComm);
}

/*
    Uses different assignment than proposed originally (using C=1 matrix distribution and then shifting).
    Instead approach that simplifies id calculation is used, namely single replication group consists of
    processes with same division remainder, but instead of consecutive groups assignment, we are doing somekind
    of shift to ensure that processes from a single dense matrix replication group will process unique blocks of matrix:
    Example for P=18, C=3:
        Replication groups:
            0 - [0, 6, 12]
            1 - [3, 9, 15]
            2 - [1, 7, 13]
            3 - [4, 10, 16]
            4 - [2, 8, 14]
            5 - [5, 11, 17]
*/
template <>
SparseMatrixReplicationGroup SparseMatrixReplicationGroup::ofProcess<Algorithm::InnerABC>(int processId,
                                                                                          int numProcesses,
                                                                                          int numReplicationGroups,
                                                                                          int replicationGroupSize) {
    int numReplicationLayers = replicationGroupSize;
    int numShifts = numReplicationGroups / replicationGroupSize;  // shifts required for a single multiplication
    int denseLayerId = processId % numReplicationLayers;
    int denseGroupId = processId / replicationGroupSize;

    int localRgId = (denseLayerId * numShifts) + (denseGroupId % numShifts);
    int localRgGlobalId = SparseMatrixReplicationGroup::getGlobalId(localRgId, numReplicationGroups);
    int localLeaderId = processId % numReplicationGroups;

    MPI_Comm localComm;
    MPI_Comm_split(MPI_COMM_WORLD, localRgGlobalId, processId, &localComm);

    int predRgId = (localRgId - 1 + numReplicationGroups) % numReplicationGroups;
    int predRgGlobalId = SparseMatrixReplicationGroup::getGlobalId(predRgId, numReplicationGroups);
    int predRgLeaderId = (predRgId / numShifts) + (predRgId % numShifts) * replicationGroupSize;
    MPI_Comm predInterComm;

    int succRgId = (localRgId + 1) % numReplicationGroups;
    int succRgGlobalId = SparseMatrixReplicationGroup::getGlobalId(succRgId, numReplicationGroups);
    int succRgLeaderId = (succRgId / numShifts) + (succRgId % numShifts) * replicationGroupSize;
    MPI_Comm succInterComm;

    if (localRgId % 2 == 0) {
        MPI_Intercomm_create(localComm, INTERNAL_LEADER_ID, MPI_COMM_WORLD, succRgLeaderId, localRgGlobalId,
                             &succInterComm);
        if (processId == localLeaderId) {
            MPI_Intercomm_create(MPI_COMM_SELF, INTERNAL_LEADER_ID, MPI_COMM_WORLD, predRgLeaderId, predRgGlobalId,
                                 &predInterComm);
        }
    } else {
        if (processId == localLeaderId) {
            MPI_Intercomm_create(MPI_COMM_SELF, INTERNAL_LEADER_ID, MPI_COMM_WORLD, predRgLeaderId, predRgGlobalId,
                                 &predInterComm);
        }
        MPI_Intercomm_create(localComm, INTERNAL_LEADER_ID, MPI_COMM_WORLD, succRgLeaderId, localRgGlobalId,
                             &succInterComm);
    }

    return SparseMatrixReplicationGroup(localRgId, replicationGroupSize, localLeaderId, localComm, predInterComm,
                                        succInterComm);
}

DenseMatrixReplicationGroup DenseMatrixReplicationGroup::ofProcess(int processId, int numProcesses,
                                                                   int numReplicationGroups, int replicationGroupSize,
                                                                   Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::ColumnA:
            return DenseMatrixReplicationGroup::ofProcess<Algorithm::ColumnA>(
                processId, numProcesses, numReplicationGroups, replicationGroupSize);
        default:
        case Algorithm::InnerABC:
            return DenseMatrixReplicationGroup::ofProcess<Algorithm::InnerABC>(
                processId, numProcesses, numReplicationGroups, replicationGroupSize);
    }
}

SparseMatrixReplicationGroup SparseMatrixReplicationGroup::ofProcess(int processId, int numProcesses,
                                                                     int numReplicationGroups, int replicationGroupSize,
                                                                     Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::ColumnA:
            return SparseMatrixReplicationGroup::ofProcess<Algorithm::ColumnA>(
                processId, numProcesses, numReplicationGroups, replicationGroupSize);
        default:
        case Algorithm::InnerABC:
            return SparseMatrixReplicationGroup::ofProcess<Algorithm::InnerABC>(
                processId, numProcesses, numReplicationGroups, replicationGroupSize);
    }
}