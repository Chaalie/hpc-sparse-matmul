#include "common.h"
#include "matrix.h"
#include "communication.h"
#include "multiplication.h"

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
        SparseMatrix fullA = SparseMatrix::fromFile(options.sparseMatrixFile);

        // Send to each process its fragment of the matrix
        std::vector<communication::Request> requests(numProcesses - 1);
        for (int p = 0; p < numProcesses; p++) {
            auto frag = SparseMatrix::fragmentOfProcess(env, p);
            auto matFrag = std::make_shared<SparseMatrix>(fullA.maskSubMatrix(frag));
            if (p == env.localId) {
                A = std::move(*matFrag);
            } else {
                requests[p - 1] = communication::Isend<SparseMatrix>(matFrag, p);
            }
        }
        // std::vector<MPI_Status> statuses(numProcesses - 1);
        // MPI_Waitall(requests.size(), requests.data(), statuses.data());
        for (auto &req: requests) {
            MPI_Wait(req.mpi_request.get(), MPI_STATUS_IGNORE);;
        }
    } else {
        // Get the matrix fragment from coordinator
        communication::Recv<SparseMatrix>(A, COORDINATOR_PROCESS_ID);
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    auto frag = DenseMatrix::fragmentOfProcess(env, processId);
    MatrixIndex s, e;
    std::tie(s, e) = frag;
    DenseMatrix B = DenseMatrix::generate(frag, options.denseMatrixSeed);

    DenseMatrix C = multiply(env, A, B, options.multiplicationExponent);

    if (options.printMatrix) {
        if (env.localIsCoordinator) {
            DenseMatrix resultMatrix = DenseMatrix::blank(env.matrixDimension, env.matrixDimension);

            for (int p = 0; p < numProcesses; p++) {
                auto frag = DenseMatrix::fragmentOfProcess(env, p);
                DenseMatrix fragC;

                if (p == env.localId) {
                    fragC = std::move(C);
                } else {
                    fragC = communication::Recv<DenseMatrix>(p);
                }

                MatrixIndex begin, end;
                std::tie(begin, end) = frag;
                for (int r = begin.row; r <= end.row; r++) {
                    for (int c = begin.col; c <= end.col; c++) {
                        resultMatrix.values[r][c] = fragC.values[r - begin.row][c - begin.col];
                    }
                }
            }

            resultMatrix.print();
        } else {
            communication::Send<DenseMatrix>(C, COORDINATOR_PROCESS_ID);
        }
    }

    MPI_Finalize();
    return 0;
}