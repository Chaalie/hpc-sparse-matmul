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
    void Send(T& data, int destProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    void Isend(T& data, int destProcessId, MPI_Request& req, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    void Recv(T& data, int srcProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    T Recv(int srcProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    void Irecv(T& data, int srcProcessId, MPI_Request& req, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);
};

#endif /* __COMMUNICATION_H__ */