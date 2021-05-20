#include "common.h"

#include <set>
#include <map>

enum OptionType {
    POSITIONAL = 0,
    NAMED,
    FLAG
};

enum OptionRequired {
    REQUIRED = 0,
    OPTIONAL
};

class OptionBase {
public:
    OptionRequired required;
    OptionType type;
    std::string valueName;
    std::string description;

    virtual void parse(const std::string &arg) const = 0;

    OptionBase(OptionRequired required, OptionType type, std::string valueName, std::string description) :
        required(required),
        type(type),
        valueName(valueName),
        description(description) {}
};

template <typename T>
class Option : public OptionBase {
private:
    T* dest;

public:
    Option(OptionRequired required, OptionType type, std::string valueName, std::string description, T* dest)
        : OptionBase(required, type, valueName, description), dest(dest) {}
    
    void parse(const std::string &arg) const;
};

template <>
void Option<std::string>::parse(const std::string &arg) const { *dest = arg; }

template <>
void Option<int>::parse(const std::string &arg) const { *dest = std::atoi(arg.c_str()); }

template <>
void Option<bool>::parse(const std::string &arg) const { *dest = true; }

void printUsage() {
    std::cout << "Usage" << std::endl;
}

ProgramOptions::ProgramOptions(int argc, char* argv[]) {
    std::string sparseMatrixFile;
    int denseMatrixSeed;
    int numReplicationGroups;
    int multiplicationExponent;
    bool useInnerAlgorithm = false;
    bool printMatrix = false;
    bool printGreaterEqual = false;
    int printGreaterEqualValue;

    const std::map<std::string, OptionBase*> supportedOptions {
        {"-f", new Option<std::string>(REQUIRED,  NAMED, "sparse_matrix_file",    "", &sparseMatrixFile) },
        {"-s",         new Option<int>(REQUIRED,  NAMED, "seed_for_dense_matrix", "", &denseMatrixSeed) },
        {"-c",         new Option<int>(REQUIRED,  NAMED, "repl_group_size",       "", &numReplicationGroups) },
        {"-e",         new Option<int>(REQUIRED,  NAMED, "exponent",              "", &multiplicationExponent) },
        {"-g",         new Option<int>(OPTIONAL,  NAMED, "ge_value",              "", &printGreaterEqualValue) },
        {"-v",        new Option<bool>(OPTIONAL,  FLAG,  "",                      "", &printMatrix) },
        {"-i",        new Option<bool>(OPTIONAL,  FLAG,  "",                      "", &useInnerAlgorithm) },
    };

    std::set<std::string> foundOptions;
    for (int i = 1; i < argc; i++) {
        std::string optionName = argv[i];
        auto optIt = supportedOptions.find(optionName);
        if (optIt != supportedOptions.end()) {
            if (foundOptions.find(optionName) != foundOptions.end()) {
                std::cout << "Duplicated option: " << optionName << std::endl;
                printUsage();
                exit(1);
            } else {
                foundOptions.insert(optionName);

                if (optIt->second->type == FLAG) {
                    optIt->second->parse("");
                } else {
                    optIt->second->parse(argv[++i]);
                }
            }
        } else {
            std::cout << "Unrecognized option: " << optionName << std::endl;
            printUsage();
            exit(1);
        }
    }

    for (auto it: supportedOptions) {
        if (it.second->required == REQUIRED && foundOptions.find(it.first) == foundOptions.end()) {
            std::cout << "Missing required option: " << it.first << std::endl;
            printUsage();
            exit(1);
        }
    }

    if (foundOptions.find("-g") != foundOptions.end()){
        printGreaterEqual = true;
    }

    this->sparseMatrixFile = sparseMatrixFile;
    this->denseMatrixSeed = denseMatrixSeed;
    this->numReplicationGroups = numReplicationGroups;
    this->multiplicationExponent = multiplicationExponent;
    this->useInnerAlgorithm = useInnerAlgorithm;
    this->printMatrix = printMatrix;
    this->printGreaterEqual = printGreaterEqual;
    this->printGreaterEqualValue = printGreaterEqualValue;
}

Environment::Environment(int localId, int numProcesses, int matrixDimension, ProgramOptions options) : localId(localId),
                                                                                                       numProcesses(numProcesses),
                                                                                                       numReplicationGroups(options.numReplicationGroups),
                                                                                                       matrixDimension(matrixDimension),
                                                                                                       localIsCoordinator(isCoordinator(localId)) {}

ReplicationGroup::ReplicationGroup(int id, int numProcesses, int numReplicationGroups) : id(id) {
    this->firstProcessId =
        this->id * (numProcesses / numReplicationGroups)
      + std::min(this->id, numProcesses % numReplicationGroups);
    this->size = std::min(
        numProcesses - this->firstProcessId,
        numProcesses / numReplicationGroups + (this->id < numProcesses % numReplicationGroups)
    );
    this->lastProcessId = this->firstProcessId + this->size - 1;
}

int ReplicationGroup::internalProcessId(int processId) {
    if (this->firstProcessId <= processId && processId <= this->lastProcessId) {
        return processId - this->firstProcessId;
    } else {
        return -1;
    }
}

ReplicationGroup ReplicationGroup::ofId(Environment& env, int groupId) {
    return ReplicationGroup(groupId, env.numProcesses, env.numReplicationGroups);
}

ReplicationGroup ReplicationGroup::ofProcess(Environment& env, int processId) {
    int baseGroupSize = env.numProcesses / env.numReplicationGroups;
    int numOfBiggerGroups = env.numProcesses % env.numReplicationGroups;
    int processesInBiggerGroups = numOfBiggerGroups * (baseGroupSize + 1);

    int groupId;
    if (processId < processesInBiggerGroups) {
        groupId = processId / (baseGroupSize + 1);
    } else {
        groupId = numOfBiggerGroups + (processId - processesInBiggerGroups) / baseGroupSize;
    }
    return ReplicationGroup::ofId(env, groupId);
}

int getSparseMatrixFirstColumn(Environment& env, int processId) {
    if (processId >= env.numProcesses) {
        return env.matrixDimension + 1;
    }

    ReplicationGroup rg = ReplicationGroup::ofProcess(env, processId);
    int internalProcessId = rg.internalProcessId(processId);
    assert(internalProcessId >= 0);

    int baseColumnsPerProcess = env.matrixDimension / rg.size;
    int firstColumn =
        internalProcessId * baseColumnsPerProcess
      + std::min(internalProcessId, env.matrixDimension % rg.size);

    return firstColumn;
}

MatrixRange getSparseMatrixRangeOfProcess(Environment& env, int processId) {
    ReplicationGroup rg = ReplicationGroup::ofProcess(env, processId);
    int internalProcessId = rg.internalProcessId(processId);
    assert(internalProcessId >= 0);

    int baseColumnsPerProcess = env.matrixDimension / rg.size;
    int colBeginIncl =
        internalProcessId * baseColumnsPerProcess
      + std::min(internalProcessId, env.matrixDimension % rg.size);
    int colEndExcl = colBeginIncl + baseColumnsPerProcess + (internalProcessId < env.matrixDimension % rg.size);

    return {
        { 0, colBeginIncl },
        { env.matrixDimension - 1, colEndExcl - 1 }
    };
}

int getDenseMatrixFirstColumn(Environment& env, int processId) {
    return 1;
}

MatrixRange getDenseMatrixRangeOfProcess(Environment& env, int processId) {
    int colBeginIncl = getDenseMatrixFirstColumn(env, processId);
    int colEndExcl = getDenseMatrixFirstColumn(env, processId + 1); 

    return {
        { 0, colBeginIncl },
        { env.matrixDimension - 1, colEndExcl - 1 }
    };
}