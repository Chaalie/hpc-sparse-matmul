#ifndef __COMMON_H__
#define __COMMON_H__

#include "communication.h"

#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <functional>

class ProgramOptions {
   public:
    std::string sparseMatrixFile;
    int denseMatrixSeed;
    int numReplicationGroups;
    int multiplicationExponent;
    bool useInnerAlgorithm;
    bool printMatrix;
    bool printGreaterEqual;
    int printGreaterEqualValue;

    ProgramOptions(int argc, char* argv[]);

    void print() {
        std::cout << "sparseMatrixFile: " << sparseMatrixFile << std::endl;
        std::cout << "denseMatrixSeed: " << denseMatrixSeed << std::endl;
        std::cout << "numReplicationGroups: " << numReplicationGroups << std::endl;
        std::cout << "multiplicationExponent: " << multiplicationExponent << std::endl;
        std::cout << "useInnerAlgorithm: " << std::string(useInnerAlgorithm ? "True" : "False") << std::endl;
        std::cout << "printMatrix: " << std::string(printMatrix ? "True" : "False") << std::endl;
        std::cout << "printGreaterEqual: " << std::string(printGreaterEqual ? "True" : "False") << std::endl;
        std::cout << "printGreaterEqualValue: " << printGreaterEqualValue << std::endl;
    }
};

class Environment {
public:
    const int localId;
    const int numProcesses;
    const int numReplicationGroups;
    const int matrixDimension;
    const int localIsCoordinator;

    Environment(int localId, int numProcesses, int matrixDimension, ProgramOptions options);
};

class ReplicationGroup {
private:
    ReplicationGroup(int id, int numProcess, int numReplicationGroups);

public:
    int id;
    int size;
    int firstProcessId;
    int lastProcessId;

    int internalProcessId(int processId);
    int successorOf(int processId);
    int predecessorOf(int processId);
    static ReplicationGroup ofId(Environment& env, int groupId);
    static ReplicationGroup ofProcess(Environment& env, int processId);
};

#endif /* __COMMON_H__ */