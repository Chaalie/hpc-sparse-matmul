#ifndef __REPLICATION_GROUP_H__
#define __REPLICATION_GROUP_H__

#include <mpi.h>

#include <cassert>
#include <string>

#include "common.h"

class ReplicationGroup {
public:
    const int id = -1;
    const int size = -1;
    const int leaderId = -1;
    const MPI_Comm internalComm = MPI_COMM_NULL;

    bool isLeader(const int id) const { return id == leaderId; }

    ReplicationGroup() {}

protected:
    ReplicationGroup(int id, int size, int leaderId, MPI_Comm internalComm)
        : id(id), size(size), leaderId(leaderId), internalComm(internalComm) {}
};

class DenseMatrixReplicationGroup : public ReplicationGroup {
public:
    // template <Algorithm A>
    // static DenseMatrixReplicationGroup ofId(int id, int numProcesses, int numReplicationGroups,
    //                                         int replicationGroupSize);

    static DenseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                  int replicationGroupSize, Algorithm algorithm);

    // Id used for constructing new MPI communicators, thus needs to be unique across all replication groups
    static int getGlobalId(int id, int numGroups) {
        assert(id < numGroups);
        return id;
    }

private:
    using ReplicationGroup::ReplicationGroup;
    
    template <Algorithm A>
    static DenseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                 int replicationGroupSize);
};

class SparseMatrixReplicationGroup : public ReplicationGroup {
public:
    const MPI_Comm predInterComm = MPI_COMM_NULL;  // inter communicator for previous replication group
    const MPI_Comm succInterComm = MPI_COMM_NULL;  // inter communicator for next replication group

    // template <Algorithm A>
    // static SparseMatrixReplicationGroup ofId(int id, int numProcesses, int numReplicationGroups,
    //                                          int replicationGroupSize);

    static SparseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                  int replicationGroupSize, Algorithm algorithm);

    // Id used for constructing new MPI communicators, thus needs to be unique across all replication groups
    static int getGlobalId(int id, int numGroups) {
        assert(id < numGroups);
        return numGroups + id;
    }

private:
    SparseMatrixReplicationGroup(int id, int size, int leaderId, MPI_Comm internalComm, MPI_Comm predInterComm,
                                 MPI_Comm succInterComm)
        : ReplicationGroup(id, size, leaderId, internalComm),
          predInterComm(predInterComm),
          succInterComm(succInterComm) {}

    template <Algorithm A>
    static SparseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                  int replicationGroupSize);
};

#endif /* __REPLICATION_GROUP_H__ */