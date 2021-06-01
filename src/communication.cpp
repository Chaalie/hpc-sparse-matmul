#include "communication.h"

#include "common.h"

#include <mpi.h>
#include <fstream>
#include <cassert>
#include <memory>

using namespace communication;

int getMatrixDimension(int processId, std::string& sparseMatrixFileName) {
    int dimension;

    if (isMainLeader(processId)) {
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