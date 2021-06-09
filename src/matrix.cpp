#include <mpi.h>

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "../densematgen.h"
#include "common.h"
#include "context.h"
#include "matrix.h"
#include "mpi_helpers.h"

std::ostream& operator<<(std::ostream& os, const MatrixIndex& mIdx) {
    os << "("
       << "row: " << mIdx.row << ", "
       << "col: " << mIdx.col << ")";
    return os;
}

SparseMatrix SparseMatrix::fromFile(std::string& otherFileName) {
    std::ifstream otherFile(otherFileName);

    int rows, columns, nonZerosCount, nonZerosPerRow;

    otherFile >> rows >> columns >> nonZerosCount >> nonZerosPerRow;
    assert(rows == columns);

    std::vector<double> nonZeros(nonZerosCount);
    for (uint i = 0; i < nonZeros.size(); i++) {
        otherFile >> nonZeros[i];
    }

    std::vector<int> rowIdx(rows + 1);
    for (uint i = 0; i < rowIdx.size(); i++) {
        otherFile >> rowIdx[i];
    }

    std::vector<int> colIdx(nonZerosCount);
    for (uint i = 0; i < colIdx.size(); i++) {
        otherFile >> colIdx[i];
    }

    otherFile.close();
    return SparseMatrix({rows, columns}, nonZeros, rowIdx, colIdx);
}

/* Returns an original other filled with zeros besides provided subother. */
SparseMatrix SparseMatrix::maskSubMatrix(MatrixFragment& fragment) {
    std::vector<double> newValues;
    std::vector<int> newRowIdx(this->rowIdx.size());
    std::vector<int> newColIdx;
    int newValuesCount = 0;

    MatrixIndex maskStart, maskEnd;
    std::tie(maskStart, maskEnd) = fragment;

    newRowIdx[0] = 0;
    for (int r = 0; r < this->dimension.row; r++) {
        if (maskStart.row <= r && r < maskEnd.row) {
            int rowStart = this->rowIdx[r];
            int rowEnd = this->rowIdx[r + 1];

            for (int i = rowStart; i < rowEnd; i++) {
                int c = this->colIdx[i];
                if (maskStart.col <= c && c < maskEnd.col) {
                    newValuesCount++;
                    newValues.push_back(this->values[i]);
                    newColIdx.push_back(c);
                }
            }
        }
        newRowIdx[r + 1] = newValuesCount;
    }

    newValues.shrink_to_fit();
    newColIdx.shrink_to_fit();
    return SparseMatrix(this->dimension, newValues, newRowIdx, newColIdx);
}

template <>
PackedData pack<SparseMatrix>(SparseMatrix& matrix, MPI_Comm comm) {
    PackedData buf;
    int size, pos = 0, packSize;

    // pack @matrix.values
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = matrix.values.size();
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    MPI_Pack_size(matrix.values.size(), MPI_DOUBLE, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(matrix.values.data(), matrix.values.size(), MPI_DOUBLE, buf.data(), buf.size(), &pos, comm);

    // pack @matrix.rowIdx
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = matrix.rowIdx.size();
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    MPI_Pack_size(matrix.rowIdx.size(), MPI_INT, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(matrix.rowIdx.data(), matrix.rowIdx.size(), MPI_INT, buf.data(), buf.size(), &pos, comm);

    // pack @matrix.colIdx
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = matrix.colIdx.size();
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    MPI_Pack_size(matrix.colIdx.size(), MPI_INT, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(matrix.colIdx.data(), matrix.colIdx.size(), MPI_INT, buf.data(), buf.size(), &pos, comm);

    return buf;
}

template <>
SparseMatrix unpack<SparseMatrix>(char* buf, int size, MPI_Comm comm) {
    int pos = 0;

    // unpack @this->values
    int valuesSize;
    MPI_Unpack(buf, size, &pos, &valuesSize, 1, MPI_INT, comm);
    std::vector<double> values(valuesSize);
    MPI_Unpack(buf, size, &pos, values.data(), valuesSize, MPI_DOUBLE, comm);

    // unpack @this->rowIdx
    int rowIdxSize;
    MPI_Unpack(buf, size, &pos, &rowIdxSize, 1, MPI_INT, comm);
    std::vector<int> rowIdx(rowIdxSize);
    MPI_Unpack(buf, size, &pos, rowIdx.data(), rowIdxSize, MPI_INT, comm);

    // unpack @this->colIdx
    int colIdxSize;
    MPI_Unpack(buf, size, &pos, &colIdxSize, 1, MPI_INT, comm);
    std::vector<int> colIdx(colIdxSize);
    MPI_Unpack(buf, size, &pos, colIdx.data(), colIdxSize, MPI_INT, comm);

    int rows, columns = rows = rowIdxSize - 1;
    return SparseMatrix({rows, columns}, values, rowIdx, colIdx);
}

template <>
SparseMatrix unpack<SparseMatrix>(PackedData& packedData, MPI_Comm comm) {
    return unpack<SparseMatrix>(packedData.data(), packedData.size(), comm);
};

void SparseMatrix::print(int verbosity) {
    std::cout << this->dimension.row << " " << this->dimension.col << "\n";
    if (verbosity <= 0) {
        this->printShort();
    } else {
        this->printFull();
    }
}

void SparseMatrix::printFull() {
    int vIdx = 0;
    for (int r = 0; r < this->dimension.row; r++) {
        int rowStart = rowIdx[r];
        int rowEnd = rowIdx[r + 1];

        for (int c = 0; c < this->dimension.col; c++) {
            double value = (rowStart <= vIdx && vIdx < rowEnd && this->colIdx[vIdx] == c) ? this->values[vIdx++] : 0.0;
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}

void SparseMatrix::printShort() {
    std::cout << "V: ";
    for (auto& v : this->values) {
        std::cout << std::fixed << v << " ";
    }
    std::cout << "\n";

    std::cout << "C: ";
    for (auto& v : this->colIdx) {
        std::cout << std::fixed << v << " ";
    }
    std::cout << "\n";

    std::cout << "R: ";
    for (auto& v : this->rowIdx) {
        std::cout << std::fixed << v << " ";
    }
    std::cout << "\n";
}

void SparseMatrix::join(SparseMatrix&& matrix) {
    SparseMatrix m = std::move(matrix);
    assert(dimension.col == m.dimension.col);
    assert(dimension.row == m.dimension.row);

    SparseMatrix* left = this;
    SparseMatrix* right = &m;
    std::vector<double> values(left->values.size() + right->values.size());
    std::vector<int> colIdx(left->colIdx.size() + right->colIdx.size());
    std::vector<int> rowIdx(this->dimension.row + 1);

    int idx = 0;
    int lIdx = 0;
    int rIdx = 0;
    int row = 0;
    while (row != this->dimension.row) {
        int goLeft = (left->rowIdx[row] <= lIdx && lIdx < left->rowIdx[row + 1]);
        int goRight = (right->rowIdx[row] <= rIdx && rIdx < right->rowIdx[row + 1]);
        if (goLeft && goRight) {
            if (left->colIdx[lIdx] < right->colIdx[rIdx]) {
                goRight = false;
            } else if (left->colIdx[lIdx] > right->colIdx[rIdx]) {
                goLeft = false;
            } else {
                throw "Matrices overlaps";
            }
        }

        if (goLeft) {
            values[idx] = left->values[lIdx];
            colIdx[idx++] = left->colIdx[lIdx++];
        } else if (goRight) {
            values[idx] = right->values[rIdx];
            colIdx[idx++] = right->colIdx[rIdx++];
        } else {
            row++;
            rowIdx[row] = idx;
        }
    }

    this->values = std::move(values);
    this->colIdx = std::move(colIdx);
    this->rowIdx = std::move(rowIdx);
}

SparseMatrix SparseMatrix::blank(MatrixDimension dimension) {
    std::vector<double> values;
    std::vector<int> rowIdx(dimension.row + 1, 0);
    std::vector<int> colIdx;
    return SparseMatrix(dimension, values, rowIdx, colIdx);
}

void SparseMatrix::Iterator::adjustRowIdx() {
    while (curRowIdx < (int)other->rowIdx.size() &&
           !(other->rowIdx[curRowIdx] <= curValueIdx && curValueIdx < other->rowIdx[curRowIdx + 1])) {
        curRowIdx++;
    }
}

SparseMatrix::Iterator::Iterator(SparseMatrix* other, int curValueIdx) noexcept
    : other(other), curRowIdx(0), curValueIdx(curValueIdx) {
    adjustRowIdx();
}

SparseMatrix::Field SparseMatrix::Iterator::operator*() const {
    return {{this->curRowIdx, this->other->colIdx[this->curValueIdx]}, this->other->values[this->curValueIdx]};
}

SparseMatrix::Iterator& SparseMatrix::Iterator::operator++() {
    if (this->curValueIdx < (int)this->other->values.size()) {
        this->curValueIdx++;
        adjustRowIdx();
    }
    return *this;
}

SparseMatrix::Iterator SparseMatrix::Iterator::operator++(int) {
    Iterator tmp = *this;
    ++*this;
    return tmp;
}

bool operator==(const SparseMatrix::Iterator& a, const SparseMatrix::Iterator& b) {
    return a.other == b.other && (a.curValueIdx == b.curValueIdx || (a.curValueIdx >= (int)a.other->values.size() &&
                                                                     b.curValueIdx >= (int)b.other->values.size()));
};

bool operator!=(const SparseMatrix::Iterator& a, const SparseMatrix::Iterator& b) { return !(a == b); };

SparseMatrix::Iterator SparseMatrix::begin() { return SparseMatrix::Iterator(this, 0); }

SparseMatrix::Iterator SparseMatrix::end() { return SparseMatrix::Iterator(this, this->values.size()); }

DenseMatrix DenseMatrix::blank(MatrixDimension dimension) {
    std::vector<double> data(dimension.row * dimension.col, 0.0);
    return DenseMatrix(dimension, data);
}

DenseMatrix DenseMatrix::generate(MatrixFragment& frag, int seed) {
    MatrixIndex start, end;
    std::tie(start, end) = frag;
    int numColumns = end.col - start.col;
    int numRows = end.row - start.row;
    assert(numColumns > 0);
    assert(numRows > 0);

    std::vector<double> data(numRows * numColumns);
    int idx = 0;
    for (int c = start.col; c < end.col; c++) {
        for (int r = start.row; r < end.row; r++) {
            data[idx++] = generate_double(seed, r, c);
        }
    }

    return DenseMatrix({numRows, numColumns}, data);
}

template <>
PackedData pack<DenseMatrix>(DenseMatrix& matrix, MPI_Comm comm) {
    PackedData buf;
    int size, pos = 0;

    // pack number of columns
    MPI_Pack_size(2, MPI_INT, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(&matrix.dimension, 2, MPI_INT, buf.data(), buf.size(), &pos, comm);

    // pack each row
    MPI_Pack_size(matrix.data.size(), MPI_DOUBLE, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(matrix.data.data(), matrix.data.size(), MPI_DOUBLE, buf.data(), buf.size(), &pos, comm);

    return buf;
}

template <>
DenseMatrix unpack<DenseMatrix>(char* buf, int size, MPI_Comm comm) {
    int pos = 0;

    // unpack dimension
    int dimension[2];
    MPI_Unpack(buf, size, &pos, dimension, 2, MPI_INT, comm);
    int rows = dimension[0], columns = dimension[1];

    // unpack data
    int dataSize = rows * columns;
    std::vector<double> data(dataSize);
    MPI_Unpack(buf, size, &pos, data.data(), dataSize, MPI_DOUBLE, comm);

    return DenseMatrix({rows, columns}, data);
}

template <>
DenseMatrix unpack<DenseMatrix>(PackedData& packedData, MPI_Comm comm) {
    return unpack<DenseMatrix>(packedData.data(), packedData.size(), comm);
};

void DenseMatrix::join(DenseMatrix&& matrix) {
    DenseMatrix m = std::move(matrix);
    this->dimension.col += matrix.dimension.col;
    this->data.insert(this->data.end(), std::make_move_iterator(m.data.begin()), std::make_move_iterator(m.data.end()));
}

int DenseMatrix::countGE(MatrixFragment fragment, double geValue) {
    MatrixIndex start, end;
    std::tie(start, end) = fragment;

    int ret = 0;
    for (int c = start.col; c < end.col; c++) {
        for (int r = start.row; r < end.row; r++) {
            ret += ((*this)(r, c) >= geValue);
        }
    }
    return ret;
}

double& DenseMatrix::operator()(int rowIdx, int colIdx) {
    int idx = colIdx * this->dimension.row + rowIdx;
    return this->data[idx];
}

void DenseMatrix::print(int) {
    std::cout << this->dimension.row << " " << this->dimension.col << "\n";
    for (int r = 0; r < this->dimension.row; r++) {
        for (int c = 0; c < this->dimension.col; c++) {
            std::cout << std::setprecision(5) << std::fixed << " " << (*this)(r, c);
        }
        std::cout << "\n";
    }
}