#include "common.h"

#include <set>
#include <map>
#include <memory>
#include <functional>

class OptionBase {
public:
    bool required;
    bool isFlag;
    std::string valueName;
    std::string description;

    virtual void parse(const std::string &arg) const = 0;

    OptionBase(bool required, bool isFlag, std::string valueName, std::string description) :
        required(required),
        isFlag(isFlag),
        valueName(valueName),
        description(description) {}
};

template <typename T>
class Option : public OptionBase {
private:
    T* dest;

public:
    Option(bool required, bool isFlag, std::string valueName, std::string description, T* dest)
        : OptionBase(required, isFlag, valueName, description), dest(dest) {};
    
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

InputOptions InputOptions::parse(int argc, char* argv[]) {
    std::string sparseMatrixFile;
    int denseMatrixSeed;
    int replicationGroupSize;
    int multiplicationExponent;
    bool useInnerAlgorithm = false;
    bool printMatrix = false;
    bool printGreaterEqual = false;
    int printGreaterEqualValue;

    const std::map<std::string, OptionBase*> supportedOptions {
        {"-f", new Option<std::string>(true,  false, "sparse_matrix_file",    "", &sparseMatrixFile) },
        {"-s",         new Option<int>(true,  false, "seed_for_dense_matrix", "", &denseMatrixSeed) },
        {"-c",         new Option<int>(true,  false, "repl_group_size",       "", &replicationGroupSize) },
        {"-e",         new Option<int>(true,  false, "exponent",              "", &multiplicationExponent) },
        {"-g",         new Option<int>(false, false, "ge_value",              "", &printGreaterEqualValue) },
        {"-v",        new Option<bool>(false, true,  "",                      "", &printMatrix) },
        {"-i",        new Option<bool>(false, true,  "",                      "", &useInnerAlgorithm) },
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

                if (optIt->second->isFlag) {
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
        if (it.second->required && foundOptions.find(it.first) == foundOptions.end()) {
            std::cout << "Missing required option: " << it.first << std::endl;
            printUsage();
            exit(1);
        }
    }

    if (foundOptions.find("-g") != foundOptions.end()){
        printGreaterEqual = true;
    }

    return InputOptions(
        sparseMatrixFile,
        denseMatrixSeed,
        replicationGroupSize,
        multiplicationExponent,
        useInnerAlgorithm,
        printMatrix,
        printGreaterEqual,
        printGreaterEqualValue
    );
}