#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "common.h"
#include "communication.h"

#include <mpi.h>

#include <tuple>
#include <vector>
#include <memory>
#include <fstream>
#include <cassert>
#include <iostream>

struct MatrixIndex {
    int row;
    int col;

    friend std::ostream& operator<< (std::ostream &out, const MatrixIndex& mIdx);
};

typedef std::tuple<MatrixIndex, MatrixIndex> MatrixFragment;

class SparseMatrix {
public:
    int dim;
    std::vector<double> values;
    std::vector<int> rowIdx;
    std::vector<int> colIdx;

    SparseMatrix();
    SparseMatrix(const SparseMatrix& other) = delete;
    SparseMatrix(SparseMatrix&& other) = default;

    SparseMatrix& operator=(const SparseMatrix& other) = delete;
    SparseMatrix& operator=(SparseMatrix&& other) = default;

    static SparseMatrix fromFile(std::string &otherFileName);

    /* Returns an original other filled with zeros besides provided subother. */
    SparseMatrix maskSubMatrix(MatrixFragment& fragment);

    PackedData pack(MPI_Comm comm = MPI_COMM_WORLD);
    static SparseMatrix unpack(PackedData& data, MPI_Comm comm = MPI_COMM_WORLD);

    void print();
    void printShort();

    static MatrixFragment fragmentOfProcess(Environment& env, int processId);

    typedef double FieldValue;
    typedef std::tuple<MatrixIndex, FieldValue> Field;

    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;

        Iterator(SparseMatrix* other, int curValueIdx) noexcept;

        Field operator*() const;
        Iterator& operator++();
        Iterator operator++(int);
        friend bool operator== (const Iterator& a, const Iterator& b);
        friend bool operator!= (const Iterator& a, const Iterator& b);

    private:
        SparseMatrix *other;
        int curRowIdx;
        int curValueIdx;

        void adjustRowIdx();
    };

    Iterator begin();
    Iterator end();

private:
    SparseMatrix(std::vector<double> values, std::vector<int> rowIdx, std::vector<int> colIdx);
};

template <>
void communication::Send<SparseMatrix>(SparseMatrix& mat, int destProcessId, int tag, MPI_Comm comm);

template <>
communication::Request communication::Isend<SparseMatrix>(SparseMatrix& mat, int destProcessId, int tag, MPI_Comm comm);

template <>
void communication::Recv<SparseMatrix>(SparseMatrix& mat, int srcProcessId, int tag, MPI_Comm comm);


class DenseMatrix {
public:
    typedef std::vector<double> RowType;

    int dim, numColumns;
    std::vector<RowType> values;

    DenseMatrix(); 
    DenseMatrix(const DenseMatrix& other) = delete;
    DenseMatrix(DenseMatrix&& other) = default;

    DenseMatrix& operator=(const DenseMatrix& other) = delete;
    DenseMatrix& operator=(DenseMatrix&& other) = default;

    static DenseMatrix blank(int numRows, int numColumns);

    static DenseMatrix generate(MatrixFragment& fragment, int seed);

    PackedData pack(MPI_Comm comm = MPI_COMM_WORLD);
    static DenseMatrix unpack(PackedData& data, MPI_Comm comm = MPI_COMM_WORLD);

    static MatrixFragment fragmentOfProcess(Environment& env, int processId);
    
    void print();

private:
    DenseMatrix(int dim, int numColumns, std::vector<RowType>& values);
};
template <>
void communication::Send<DenseMatrix>(DenseMatrix& mat, int destProcessId, int tag, MPI_Comm comm);

template <>
communication::Request communication::Isend<DenseMatrix>(DenseMatrix& mat, int destProcessId, int tag, MPI_Comm comm);

template <>
void communication::Recv<DenseMatrix>(DenseMatrix& mat, int srcProcessId, int tag, MPI_Comm comm);

#endif /* __MATRIX_H__ */