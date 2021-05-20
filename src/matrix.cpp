#include "matrix.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>

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

SparseMatrix::packed SparseMatrix::pack() {
    return { 1, 42, 2, 0, 1, 1, 0 };
}

SparseMatrix SparseMatrix::unpack(SparseMatrix::packed& data) {
    int idx = 0;
    std::vector<double> values(data[idx++]);
    for (int i = 0; i < (int)values.size(); i++) {
        values[i] = data[idx++];
    }

    std::vector<int> rowIdx(data[idx++]);
    for (int i = 0; i < (int)rowIdx.size(); i++) {
        rowIdx[i] = data[idx++];
    }

    std::vector<int> colIdx(data[idx++]);
    for (int i = 0; i < (int)colIdx.size(); i++) {
        colIdx[i] = data[idx++];
    }

    return SparseMatrix(values, rowIdx, colIdx);
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
            std::cout << std::fixed << value << " ";
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