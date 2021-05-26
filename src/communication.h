#ifndef __COMMUNICATION_H__
#define __COMMUNICATION_H__

#include <mpi.h>
#include <vector>
#include <string>
#include <memory>

const int COORDINATOR_PROCESS_ID = 0;

typedef std::vector<char> PackedData;

bool isCoordinator(int processId);
int getMatrixDimension(int processId, std::string& sparseMatrixFileName);

namespace communication {
    class Request;

    template <typename T>
    void Send(T& data, int destProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    Request Isend(T& data, int destProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    void Recv(T& data, int srcProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    T Recv(int srcProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    Request Irecv(int srcProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    class Request {
        private:
            std::unique_ptr<PackedData> dataPtr;
        public:
            std::unique_ptr<MPI_Request> mpi_request;

            Request();
            Request(PackedData& dataPtr);

        template <typename T>
        friend Request Isend(T& data, int destProcessId, int tag, MPI_Comm comm);

        template <typename T>
        friend Request Irecv(int srcProcessId, int tag, MPI_Comm comm);
    };
};

#endif /* __COMMUNICATION_H__ */