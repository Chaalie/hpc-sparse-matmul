#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "communication.h"

#include <mpi.h>

#include <tuple>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>

struct MatrixIndex {
    int row;
    int col;

    friend std::ostream& operator<< (std::ostream &out, const MatrixIndex& mIdx);
};

typedef std::tuple<MatrixIndex, MatrixIndex> MatrixRange;

class SparseMatrix {
private:
    SparseMatrix(std::vector<double> values, std::vector<int> rowIdx, std::vector<int> colIdx);

public:
    int dim;
    std::vector<double> values;
    std::vector<int> rowIdx;
    std::vector<int> colIdx;

    typedef std::vector<char> packed;

    typedef double FieldValue;
    typedef std::tuple<MatrixIndex, FieldValue> Field;

    SparseMatrix();

    static SparseMatrix fromFile(std::string &matrixFileName);

    /* Returns an original matrix filled with zeros besides provided submatrix. */
    SparseMatrix maskSubMatrix(MatrixRange& range);

    PackedData pack(MPI_Comm comm = MPI_COMM_WORLD);
    static SparseMatrix unpack(PackedData& data, MPI_Comm comm = MPI_COMM_WORLD);

    void Send(int destProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);
    void Isend(int destProcessId, MPI_Request& req, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);
    static SparseMatrix Recv(int srcProcessId, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD);

    void print();
    void printFull();

    class Iterator {
    private:
        SparseMatrix *matrix;
        int curRowIdx;
        int curValueIdx;

        void adjustRowIdx();

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;

        Iterator(SparseMatrix* matrix, int curValueIdx) noexcept;

        Field operator*() const;
        Iterator& operator++();
        Iterator operator++(int);
        friend bool operator== (const Iterator& a, const Iterator& b);
        friend bool operator!= (const Iterator& a, const Iterator& b);
    };

    Iterator begin();
    Iterator end();
};

#endif /* __MATRIX_H__ */