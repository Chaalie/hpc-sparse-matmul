#include "common.h"
#include "context.h"
#include "matrix.h"
#include "multiplication.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    ProgramOptions options = ProgramOptions::fromCommandLine(argc, argv);

    MPI_Init(&argc, &argv);

    int numProcesses, processId;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    SparseMatrix A;
    if (isMainLeader(processId)) {
        A = std::move(SparseMatrix::fromFile(options.sparseMatrixFile));
    }

    int matrixDimension = utils::initializeMatrixDimension(processId, A);
    Context ctx(processId, numProcesses, matrixDimension, options.replicationGroupSize, options.algorithm);

    A = utils::initializeSparseMatrix(ctx, A);
    DenseMatrix B = utils::initializeDenseMatrix(ctx, options.denseMatrixSeed);

    // int id = 0;
    // while (id < numProcesses) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (id == 0 && processId == 0) {
    //         B.print(0);
    //     } 
    //     id++;
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }


    //DenseMatrix B = utils::initializeDenseMatrix(ctx, options.denseMatrixSeed);
    // At this point, each member of replication group stores the same fragment of sparse and dense matrices (A and B)

    DenseMatrix C = multiply(ctx, std::move(A), std::move(B), options.multiplicationExponent);

    int id = 0;
    while (id < numProcesses) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == processId) {
            C.print(1);
        } 
        id++;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /*
    if (options.printMatrix) {
        DenseMatrix resultMatrix = gatherDenseMatrix(ctx, C, COORDINATOR_PROCESS_ID);
        if (ctx.process.isMainCoordinator()) {
            resultMatrix.print();
        }
    }

    if (options.printGreaterEqual) {
        int result = gatherCountGreaterEqual(ctx, C, options.printGreaterEqualValue, COORDINATOR_PROCESS_ID);
        if (ctx.process.isMainCoordinator()) {
            std::cout << result << std::endl;
        }
    }
    */

    MPI_Finalize();
    return 0;
}