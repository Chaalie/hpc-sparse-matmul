#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <mpi.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "common.h"
#include "mpi_helpers.h"

struct MatrixIndex {
    int row;
    int col;

    friend std::ostream& operator<<(std::ostream& out, const MatrixIndex& mIdx);
};

typedef MatrixIndex MatrixDimension;

typedef std::tuple<MatrixIndex, MatrixIndex> MatrixFragment;

class Matrix {
public:
    MatrixDimension dimension;

    Matrix() = default;
    Matrix(MatrixDimension dimension) : dimension(dimension) {}

    Matrix(const Matrix& other) = delete;
    Matrix(Matrix&& other) = default;

    Matrix& operator=(const Matrix& other) = delete;
    Matrix& operator=(Matrix&& other) = default;

    virtual void print(int verbosity = 0) = 0;
    virtual ~Matrix() = default;
};

class SparseMatrix : public Matrix {
public:
    SparseMatrix() = default;
    ~SparseMatrix() = default;

    SparseMatrix(const SparseMatrix& other) = delete;
    SparseMatrix(SparseMatrix&& other) = default;

    SparseMatrix& operator=(const SparseMatrix& other) = delete;
    SparseMatrix& operator=(SparseMatrix&& other) = default;

    static SparseMatrix fromFile(std::string& otherFileName);

    /* Returns an original other filled with zeros besides provided subother. */
    SparseMatrix maskSubMatrix(MatrixFragment& fragment);

    void join(SparseMatrix&& matrix);

    void print(int verbosity) override;

    static SparseMatrix blank(MatrixDimension dimension);

    typedef double FieldValue;
    typedef std::tuple<MatrixIndex, FieldValue> Field;

    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator(SparseMatrix* other, int curValueIdx) noexcept;

        Field operator*() const;
        Iterator& operator++();
        Iterator operator++(int);
        friend bool operator==(const Iterator& a, const Iterator& b);
        friend bool operator!=(const Iterator& a, const Iterator& b);

    private:
        SparseMatrix* other;
        int curRowIdx;
        int curValueIdx;

        void adjustRowIdx();
    };
    friend bool operator==(const Iterator& a, const Iterator& b);

    Iterator begin();
    Iterator end();

    friend PackedData pack<SparseMatrix>(SparseMatrix& matrix, MPI_Comm comm);
    friend SparseMatrix unpack<SparseMatrix>(char* buf, int size, MPI_Comm comm);

private:
    std::vector<double> values;
    std::vector<int> rowIdx;
    std::vector<int> colIdx;

    void printFull();
    void printShort();

    SparseMatrix(MatrixDimension dimension, std::vector<double>& values, std::vector<int>& rowIdx,
                 std::vector<int>& colIdx)
        : Matrix(dimension), values(std::move(values)), rowIdx(std::move(rowIdx)), colIdx(std::move(colIdx)) {}
};

class DenseMatrix : public Matrix {
public:
    DenseMatrix() = default;
    ~DenseMatrix() = default;

    DenseMatrix(const DenseMatrix& other) = delete;
    DenseMatrix(DenseMatrix&& other) = default;

    DenseMatrix& operator=(const DenseMatrix& other) = delete;
    DenseMatrix& operator=(DenseMatrix&& other) = default;

    // accessor, instead of operator[][], that is rough to implement optimally
    double& operator()(int rowIdx, int colIdx);

    void print(int verbosity) override;

    void join(DenseMatrix&& matrix);

    static DenseMatrix blank(MatrixDimension dimension);

    static DenseMatrix generate(MatrixFragment& fragment, int seed);

    friend PackedData pack<DenseMatrix>(DenseMatrix& matrix, MPI_Comm comm);
    friend DenseMatrix unpack<DenseMatrix>(char* buf, int size, MPI_Comm comm);

private:
    std::vector<double> data;

    DenseMatrix(MatrixDimension dimension, std::vector<double>& data) : Matrix(dimension), data(std::move(data)) {}
};

#endif /* __MATRIX_H__ */