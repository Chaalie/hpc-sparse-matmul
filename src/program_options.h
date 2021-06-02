#ifndef __PROGRAM_OPTIONS_H__
#define __PROGRAM_OPTIONS_H__

#include <iostream>
#include <string>

#include "common.h"

class ProgramOptions {
public:
    std::string sparseMatrixFile;
    int denseMatrixSeed;
    int replicationGroupSize;
    int multiplicationExponent;
    Algorithm algorithm;
    bool printMatrix;
    bool printGreaterEqual;
    double printGreaterEqualValue;

    static ProgramOptions fromCommandLine(int argc, char* argv[]);

    friend std::ostream& operator<<(std::ostream& os, ProgramOptions po);

private:
    ProgramOptions(std::string sparseMatrixFile, int denseMatrixSeed, int replicationGroupSize,
                   int multiplicationExponent, Algorithm algorithm, bool printMatrix, bool printGreaterEqual,
                   double printGreaterEqualValue)
        : sparseMatrixFile(sparseMatrixFile),
          denseMatrixSeed(denseMatrixSeed),
          replicationGroupSize(replicationGroupSize),
          multiplicationExponent(multiplicationExponent),
          algorithm(algorithm),
          printMatrix(printMatrix),
          printGreaterEqual(printGreaterEqual),
          printGreaterEqualValue(printGreaterEqualValue) {}
};

#endif /* __PROGRAM_OPTIONS_H__ */