#ifndef __COMMUNICATION_H__
#define __COMMUNICATION_H__

#include <mpi.h>
#include <vector>
#include <string>

const int COORDINATOR_PROCESS_ID = 0;

typedef std::vector<char> PackedData;

bool isCoordinator(int processId);
int getMatrixDimension(int processId, std::string& sparseMatrixFileName);

namespace communication {
    template <typename T>
    void Send(T& data, int destProcessId, int tag, MPI_Comm comm);

    template <typename T>
    void Isend(T& data, int destProcessId, int tag, MPI_Request& req, MPI_Comm comm);

    template <typename T>
    void Recv(T& data, int srcProcessId, int tag, MPI_Comm comm);

    template <typename T>
    void Irecv(T& data, int srcProcessId, int tag, MPI_Request& req, MPI_Comm comm);
};

#endif /* __COMMUNICATION_H__ */