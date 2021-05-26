#ifndef __REPLICATION_GROUP_H__
#define __REPLICATION_GROUP_H__

#include <mpi.h>
#include <vector>
#include <string>
#include <memory>

class ReplicationGroup {
    const MPI_Comm internalComm;
    const MPI_Comm shiftComm;

    static ReplicationGroup ofId(Environment& env, int id);
    static ofProcess()

    class Member {
        const bool isCoordinator;
    };

};

class ReplicationGroupMember {
public:
    const bool isCoordinator;
    const int replicationGroupId;
    const MPI_Comm internalComm;
    const MPI_Comm shiftComm;

};

#endif /* __REPLICATION_GROUP_H__ */