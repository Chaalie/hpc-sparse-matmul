#include "common.h"
#include "matrix.h"
#include "communication.h"

int main(int argc, char* argv[]) {
    ProgramOptions options(argc, argv);

    MPI_Init(&argc, &argv); 

    int numProcesses, processId;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    int matrixDimension = getMatrixDimension(processId, options.sparseMatrixFile);
    Environment env(processId, numProcesses, matrixDimension, options);

    // Get starting sparse matrix fragment
    SparseMatrix A;
    if (env.localIsCoordinator) {
        // Read a sparse matrix from file
        A = SparseMatrix::fromFile(options.sparseMatrixFile);

        // Send to each process its fragment of the matrix
        std::vector<MPI_Request> requests(numProcesses - 1);
        for (int i = 1; i < numProcesses; i++) {
            auto range = getSparseMatrixRangeOfProcess(env, i);
            A.maskSubMatrix(range).Isend(i, requests[i - 1]);
        }
        std::vector<MPI_Status> statuses(numProcesses - 1);
        MPI_Waitall(requests.size(), requests.data(), statuses.data());

        // Get coordinator's matrix fragment
        auto range = getSparseMatrixRangeOfProcess(env, processId);
        A = A.maskSubMatrix(range);
    } else {
        // Get the matrix fragment from coordinator
        A = SparseMatrix::Recv(COORDINATOR_PROCESS_ID);
    }

    int id = 0;
    while (id < numProcesses) {
        if (processId == id) {
            std::cout << processId << " is printing." << std::endl;
            A.printFull();
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