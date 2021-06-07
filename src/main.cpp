#include "common.h"
#include "context.h"
#include "matrix.h"
#include "multiplication.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    double startTime, initTime, mulpTime, endTime, gatherTime;

    ProgramOptions options = ProgramOptions::fromCommandLine(argc, argv);

    MPI_Init(&argc, &argv);
    startTime = MPI_Wtime();

    int numProcesses, processId;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    utils::verifyPreconditions(numProcesses, options.replicationGroupSize, options.algorithm);

    SparseMatrix A;
    if (isMainLeader(processId)) {
        A = std::move(SparseMatrix::fromFile(options.sparseMatrixFile));
    }

    int matrixDimension = utils::initializeMatrixDimension(processId, A);
    Context ctx(processId, numProcesses, matrixDimension, options.replicationGroupSize, options.algorithm);

    A = utils::initializeSparseMatrix(ctx, A);
    DenseMatrix B = utils::initializeDenseMatrix(ctx, options.denseMatrixSeed);
    initTime = MPI_Wtime();
    // At this point, each member of replication group stores the same fragment of sparse and dense matrices (A and B)

    DenseMatrix C = multiply(ctx, std::move(A), std::move(B), options.multiplicationExponent);
    mulpTime = gatherTime = MPI_Wtime();

    if (options.printMatrix) {
        DenseMatrix resultMatrix = utils::gatherDenseMatrix(ctx, C, MAIN_LEADER_ID);
        gatherTime = MPI_Wtime();
        if (ctx.process.isMainLeader()) {
            resultMatrix.print();
        }
    } else if (options.printGreaterEqual) {
        int result = utils::gatherCountGE(ctx, C, options.printGreaterEqualValue, MAIN_LEADER_ID);
        gatherTime = MPI_Wtime();
        if (ctx.process.isMainLeader()) {
            std::cout << result << std::endl;
        }
    }

    ctx.process.denseRG.freeComms();
    ctx.process.sparseRG.freeComms();
    endTime = MPI_Wtime();

    if (options.printStats && ctx.process.isMainLeader()) {
        std::cerr << std::fixed << "execution: " << endTime - startTime << "s" << std::endl;
        std::cerr << std::fixed << "init: " << initTime - startTime << "s" << std::endl;
        std::cerr << std::fixed << "multiplication: " << mulpTime - initTime << "s" << std::endl;
        std::cerr << std::fixed << "gather: " << gatherTime - mulpTime << "s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}