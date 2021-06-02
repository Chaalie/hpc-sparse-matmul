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
    MPI_Comm internalComm = MPI_COMM_NULL;

    bool isLeader(const int id) const { return id == leaderId; }

    ReplicationGroup() {}

    virtual void freeComms() = 0;

protected:
    ReplicationGroup(int id, int size, int leaderId, MPI_Comm internalComm)
        : id(id), size(size), leaderId(leaderId), internalComm(internalComm) {}
};

class DenseMatrixReplicationGroup : public ReplicationGroup {
public:
    MPI_Comm leadersComm = MPI_COMM_NULL;  // communicator for all replication groups leaders

    void freeComms();

    static DenseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                  int replicationGroupSize, Algorithm algorithm);

    // Id used for constructing new MPI communicators, thus needs to be unique across all replication groups
    static int getGlobalId(int id, int numGroups) {
        assert(id < numGroups);
        return id;
    }

private:
    DenseMatrixReplicationGroup(int id, int size, int leaderId, MPI_Comm internalComm, MPI_Comm leadersComm)
        : ReplicationGroup(id, size, leaderId, internalComm),
          leadersComm(leadersComm) {}
    
    template <Algorithm A>
    static DenseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                 int replicationGroupSize);
};

class SparseMatrixReplicationGroup : public ReplicationGroup {
public:
    MPI_Comm predInterComm = MPI_COMM_NULL;  // inter communicator for previous replication group
    MPI_Comm succInterComm = MPI_COMM_NULL;  // inter communicator for next replication group
    const int succInterLeader;

    void freeComms();

    static SparseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                  int replicationGroupSize, Algorithm algorithm);

    // Id used for constructing new MPI communicators, thus needs to be unique across all replication groups
    static int getGlobalId(int id, int numGroups) {
        assert(id < numGroups);
        return numGroups + id;
    }

private:
    SparseMatrixReplicationGroup(int id, int size, int leaderId, MPI_Comm internalComm, MPI_Comm predInterComm,
                                 MPI_Comm succInterComm, int succInterLeader)
        : ReplicationGroup(id, size, leaderId, internalComm),
          predInterComm(predInterComm),
          succInterComm(succInterComm),
          succInterLeader(succInterLeader) {}

    template <Algorithm A>
    static SparseMatrixReplicationGroup ofProcess(int processId, int numProcesses, int numReplicationGroups,
                                                  int replicationGroupSize);
};

#endif /* __REPLICATION_GROUP_H__ */