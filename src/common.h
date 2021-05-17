#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <functional>

class InputOptions
{
public:
  const std::string sparseMatrixFile;
  const int denseMatrixSeed;
  const int replicationGroupSize;
  const int multiplicationExponent;
  const bool useInnerAlgorithm;
  const bool printMatrix;
  const bool printGreaterEqual;
  const int printGreaterEqualValue;

  InputOptions(
      std::string sparseMatrixFile,
      int denseMatrixSeed,
      int replicationGroupSize,
      int multiplicationExponent,
      bool useInnerAlgorithm,
      bool printMatrix,
      bool printGreaterEqual,
      int printGreaterEqualValue) : sparseMatrixFile(sparseMatrixFile),
                                    denseMatrixSeed(denseMatrixSeed),
                                    replicationGroupSize(replicationGroupSize),
                                    multiplicationExponent(multiplicationExponent),
                                    useInnerAlgorithm(useInnerAlgorithm),
                                    printMatrix(printMatrix),
                                    printGreaterEqual(printGreaterEqual),
                                    printGreaterEqualValue(printGreaterEqualValue) {}
  
  static InputOptions parse(int argc, char* argv[]);

  void print() {
    std::cout << "sparseMatrixFile: "       << sparseMatrixFile << std::endl;
    std::cout << "denseMatrixSeed: "        << denseMatrixSeed << std::endl;
    std::cout << "replicationGroupSize: "   << replicationGroupSize << std::endl;
    std::cout << "multiplicationExponent: " << multiplicationExponent << std::endl;
    std::cout << "useInnerAlgorithm: "      << std::string(useInnerAlgorithm ? "True" : "False") << std::endl;
    std::cout << "printMatrix: "            << std::string(printMatrix ? "True" : "False") << std::endl;
    std::cout << "printGreaterEqual: "      << std::string(printGreaterEqual ? "True" : "False") << std::endl;
    std::cout << "printGreaterEqualValue: " << printGreaterEqualValue << std::endl;
  }
};

#endif /* __COMMON_H__ */