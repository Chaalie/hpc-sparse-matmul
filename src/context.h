#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#include <mpi.h>
#include <cassert>

#include "common.h"
#include "program_options.h"
#include "replication_group.h"

class Context {
public:
    const int numProcesses;
    const int numReplicationGroups;
    const int replicationGroupSize;
    const int numReplicationLayers;
    const int matrixDimension;
    const MPI_Comm globalComm = MPI_COMM_WORLD;
    const Algorithm algorithm;

    class ProcessInfo {
    public:
        int id;
        DenseMatrixReplicationGroup denseRG;
        SparseMatrixReplicationGroup sparseRG;

        bool isMainLeader() const { return ::isMainLeader(id); }

        ProcessInfo(int id, int numProcesses, int numReplicationGroups, int replicationGroupSize, Algorithm algorithm)
            : id(id),
              denseRG(DenseMatrixReplicationGroup::ofProcess(id, numProcesses, numReplicationGroups,
                                                                        replicationGroupSize, algorithm)),
              sparseRG(SparseMatrixReplicationGroup::ofProcess(id, numProcesses, numReplicationGroups,
                                                                          replicationGroupSize, algorithm)) {}
        
    };
    ProcessInfo process;

    Context(int processId, int numProcesses, int matrixDimension, int replicationGroupSize, Algorithm algorithm)
        : numProcesses(numProcesses),
          numReplicationGroups(numProcesses / replicationGroupSize),
          replicationGroupSize(replicationGroupSize),
          numReplicationLayers(replicationGroupSize),
          matrixDimension(matrixDimension),
          algorithm(algorithm),
          process(processId, numProcesses, numReplicationGroups, replicationGroupSize, algorithm) {}
};

#endif /* __CONTEXT_H__ */