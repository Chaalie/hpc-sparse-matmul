#include "matrix.h"
#include "common.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); 

    int numProcesses, processId;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    ProgramOptions options(argc, argv);
    int matrixDimension = getMatrixDimension(processId, options.sparseMatrixFile);
    Environment env(processId, numProcesses, matrixDimension, options);


    SparseMatrix A;
    if (env.localIsCoordinator) {
        A = SparseMatrix::fromFile(options.sparseMatrixFile);

        std::vector<MPI_Request> requests((numProcesses - 1) * 2);
        std::vector<MPI_Status> statuses((numProcesses - 1) * 2);
        for (int i = 1; i < numProcesses; i++) {
            auto range = getSparseMatrixRangeOfProcess(env, i);
            auto packedMatrix = A.maskSubMatrix(range).pack();
            auto data = packedMatrix.data();
            auto dataSize = packedMatrix.size();
            MPI_Isend(&dataSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[(i - 1) * 2]);
            MPI_Isend(data, dataSize, MPI_BYTE, i, 0, MPI_COMM_WORLD, &requests[(i - 1) * 2 + 1]);
        }
        MPI_Waitall((numProcesses - 1) * 2, requests.data(), statuses.data());

        auto range = getSparseMatrixRangeOfProcess(env, processId);
        A = A.maskSubMatrix(range);
    } else {
        int dataSize;
        MPI_Recv(&dataSize, 1, MPI_INT, COORDINATOR_PROCESS_ID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<char> data(dataSize);
        MPI_Recv(data.data(), dataSize, MPI_BYTE, COORDINATOR_PROCESS_ID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        A = SparseMatrix::unpack(data);
    }

    int id = 0;
    while (id < numProcesses) {
        if (processId == id) {
            std::cout << processId << " is printing." << std::endl;
            A.print();
            fflush(stdout);
        }
        id++;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /*

    auto { rows, columns } = getDenseMatrixRangeOfProcess(A.dim, numProcesses, options.replicationGroupSize, rank);
    DenseMatrix B = DenseMatrix::generateFromSeed(A.dim, options.denseMatrixSeed, rows, columns);
    */
    // TODO: Do calculations


    // TODO: gather results
    // if (options.printMatrix) {

    // } else if(options.printGreaterEqual) {
    //     int count = B.countGreateEqual(options.printGreaterEqualValue);
    // }

    MPI_Finalize();

    return 0;
}