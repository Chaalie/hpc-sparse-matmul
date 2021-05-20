#include "matrix.h"
#include "communication.h"

#include <mpi.h>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>

std::ostream& operator<<(std::ostream &out, const MatrixIndex& mIdx) {
    out << "(" << "row: " << mIdx.row << ", " << "col: " << mIdx.col << ")";
    return out;
}

SparseMatrix::SparseMatrix() : SparseMatrix({}, {}, {}) {}

SparseMatrix::SparseMatrix(std::vector<double> values, std::vector<int> rowIdx, std::vector<int> colIdx) : dim(rowIdx.size() - 1),
                                                                                                           values(std::move(values)),
                                                                                                           rowIdx(std::move(rowIdx)),
                                                                                                           colIdx(std::move(colIdx)) {}

SparseMatrix SparseMatrix::fromFile(std::string &matrixFileName) {
    std::ifstream matrixFile(matrixFileName);

    int width, height, nonZerosCount, nonZerosPerRow;

    matrixFile >> width >> height >> nonZerosCount >> nonZerosPerRow;
    assert(width == height);

    std::vector<double> nonZeros(nonZerosCount);
    for (uint i = 0; i < nonZeros.size(); i++) {
        matrixFile >> nonZeros[i];
    }

    std::vector<int> rowIdx(width + 1);
    for (uint i = 0; i < rowIdx.size(); i++) {
        matrixFile >> rowIdx[i];
    }

    std::vector<int> colIdx(nonZerosCount);
    for (uint i = 0; i < colIdx.size(); i++) {
        matrixFile >> colIdx[i];
    }

    matrixFile.close();
    return SparseMatrix(nonZeros, rowIdx, colIdx);
}

/* Returns an original matrix filled with zeros besides provided submatrix. */
SparseMatrix SparseMatrix::maskSubMatrix(MatrixRange& range) {
    std::vector<double> newValues;
    std::vector<int> newRowIdx(this->rowIdx.size());
    std::vector<int> newColIdx;
    int newValuesCount = 0;

    MatrixIndex maskStart, maskEnd;
    std::tie(maskStart, maskEnd) = range;

    newRowIdx[0] = 0;
    for (int r = 0; r < this->dim; r++) {
        if (maskStart.row <= r && r <= maskEnd.row) {
            int rowStart = this->rowIdx[r];
            int rowEnd   = this->rowIdx[r + 1];

            for (int i = rowStart; i < rowEnd; i++) {
                int c = this->colIdx[i];
                if (maskStart.col <= c && c <= maskEnd.col) {
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
    return SparseMatrix(newValues, newRowIdx, newColIdx);
}

PackedData SparseMatrix::pack(MPI_Comm comm) {
    PackedData buf;
    int size, pos = 0, packSize;

    // pack @this->values
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = this->values.size();
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    MPI_Pack_size(this->values.size(), MPI_DOUBLE, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(this->values.data(), this->values.size(), MPI_DOUBLE, buf.data(), buf.size(), &pos, comm);

    // pack @this->rowIdx
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = this->rowIdx.size();
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    MPI_Pack_size(this->rowIdx.size(), MPI_INT, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(this->rowIdx.data(), this->rowIdx.size(), MPI_INT, buf.data(), buf.size(), &pos, comm);

    // pack @this->colIdx
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = this->colIdx.size();
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    MPI_Pack_size(this->colIdx.size(), MPI_INT, comm, &size);
    buf.resize(pos + size);
    MPI_Pack(this->colIdx.data(), this->colIdx.size(), MPI_INT, buf.data(), buf.size(), &pos, comm);

    return buf;
}

SparseMatrix SparseMatrix::unpack(PackedData& buf, MPI_Comm comm) {
    int pos = 0;

    // unpack @this->values
    int valuesSize;
    MPI_Unpack(buf.data(), buf.size(), &pos, &valuesSize, 1, MPI_INT, comm);
    std::vector<double> values(valuesSize);
    MPI_Unpack(buf.data(), buf.size(), &pos, values.data(), valuesSize, MPI_DOUBLE, comm);

    // unpack @this->rowIdx
    int rowIdxSize;
    MPI_Unpack(buf.data(), buf.size(), &pos, &rowIdxSize, 1, MPI_INT, comm);
    std::vector<int> rowIdx(rowIdxSize);
    MPI_Unpack(buf.data(), buf.size(), &pos, rowIdx.data(), rowIdxSize, MPI_INT, comm);

    // unpack @this->colIdx
    int colIdxSize;
    MPI_Unpack(buf.data(), buf.size(), &pos, &colIdxSize, 1, MPI_INT, comm);
    std::vector<int> colIdx(colIdxSize);
    MPI_Unpack(buf.data(), buf.size(), &pos, colIdx.data(), colIdxSize, MPI_INT, comm);

    return SparseMatrix(values, rowIdx, colIdx);
}

void SparseMatrix::Send(int destProcessId, int tag, MPI_Comm comm) {
    PackedData data = this->pack();
    communication::Send<PackedData>(data, destProcessId, tag, comm);
}

void SparseMatrix::Isend(int destProcessId, MPI_Request& req, int tag, MPI_Comm comm) {
    PackedData data = this->pack();
    communication::Isend<PackedData>(data, destProcessId, tag, req, comm);
}

SparseMatrix SparseMatrix::Recv(int srcProcessId, int tag, MPI_Comm comm) {
    PackedData data;
    communication::Recv<PackedData>(data, srcProcessId, tag, comm);

    return SparseMatrix::unpack(data);
}

void SparseMatrix::print() {
    std::cout << "V: ";
    for (auto &v: this->values) {
        std::cout << std::fixed << v << " ";
    }
    std::cout << std::endl;

    std::cout << "C: ";
    for (auto &v: this->colIdx) {
        std::cout << std::fixed << v << " ";
    }
    std::cout << std::endl;

    std::cout << "R: ";
    for (auto &v: this->rowIdx) {
        std::cout << std::fixed << v << " ";
    }
    std::cout << std::endl;
}

void SparseMatrix::printFull() {
    int vIdx = 0;
    for (int r = 0; r < this->dim; r++) {
        int rowStart = rowIdx[r];
        int rowEnd = rowIdx [r + 1];

        for (int c = 0; c < this->dim; c++) {
            double value = (rowStart <= vIdx && vIdx < rowEnd && this->colIdx[vIdx] == c)
                ? this->values[vIdx++]
                : 0.0;
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

void SparseMatrix::Iterator::adjustRowIdx() {
    while (curRowIdx < (int)this->matrix->rowIdx.size()
        && !(this->matrix->values[curRowIdx] <= this->curValueIdx
            && this->curValueIdx < this->matrix->values[curRowIdx + 1])) {
        curRowIdx++;
    }
}

SparseMatrix::Iterator::Iterator(SparseMatrix* matrix, int curValueIdx) noexcept : matrix(matrix),
                                                            curRowIdx(0),
                                                            curValueIdx(curValueIdx) {
    adjustRowIdx();
}

SparseMatrix::Field SparseMatrix::Iterator::operator*() const {
    return {
        {
            this->curRowIdx,
            this->matrix->colIdx[this->curValueIdx]
        },
        this->matrix->values[this->curValueIdx]
    };
}

SparseMatrix::Iterator& SparseMatrix::Iterator::operator++() {
    if (this->curValueIdx < (int)this->matrix->values.size()) {
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
    return a.matrix == b.matrix
        && (a.curValueIdx == b.curValueIdx
            || (a.curValueIdx >= (int)a.matrix->values.size()
                && b.curValueIdx >= (int)b.matrix->values.size()));
};

bool operator!=(const SparseMatrix::Iterator& a, const SparseMatrix::Iterator& b) {
    return !(a == b);
};


SparseMatrix::Iterator SparseMatrix::begin() {
    return SparseMatrix::Iterator(this, 0);
}

SparseMatrix::Iterator SparseMatrix::end() {
    return SparseMatrix::Iterator(this, this->values.size());
}