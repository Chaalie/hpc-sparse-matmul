#include "communication.h"

#include <mpi.h>
#include <fstream>
#include <cassert>

using namespace communication;

template <>
void communication::Send<PackedData>(PackedData& data, int destProcessId, int tag, MPI_Comm comm) {
    MPI_Send(data.data(), data.size(), MPI_PACKED, destProcessId, tag, comm);
}

template <>
void communication::Isend<PackedData>(PackedData& data, int destProcessId, int tag, MPI_Request& req, MPI_Comm comm) {
    MPI_Isend(data.data(), data.size(), MPI_PACKED, destProcessId, tag, comm, &req);
}

template <>
void communication::Recv<PackedData>(PackedData& data, int srcProcessId, int tag, MPI_Comm comm) {
    MPI_Status status;
    MPI_Probe(srcProcessId, tag, comm, &status);

    int size;
    MPI_Get_count(&status, MPI_PACKED, &size);

    data.resize(size);
    MPI_Recv(data.data(), data.size(), MPI_PACKED, srcProcessId, tag, comm, &status);
}

bool isCoordinator(int processId) {
    return processId == COORDINATOR_PROCESS_ID;
}

int getMatrixDimension(int processId, std::string& sparseMatrixFileName) {
    int dimension;

    if (isCoordinator(processId)) {
        std::ifstream matrixFile(sparseMatrixFileName);

        int width, height;
        matrixFile >> width >> height;
        assert(width == height);

        matrixFile.close();

        dimension = width;
        MPI_Bcast(&dimension, 1, MPI_INT, processId, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast(&dimension, 1, MPI_INT, COORDINATOR_PROCESS_ID, MPI_COMM_WORLD);
    }

    return dimension;
}