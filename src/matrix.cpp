#include "matrix.h"
#include "common.h"
#include "communication.h"
#include "generator/densematgen.h"

#include <mpi.h>

#include <vector>
#include <memory>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <iostream>

std::ostream& operator<<(std::ostream &out, const MatrixIndex& mIdx) {
    out << "(" << "row: " << mIdx.row << ", " << "col: " << mIdx.col << ")";
    return out;
}

SparseMatrix::SparseMatrix() {}

SparseMatrix::SparseMatrix(std::vector<double> values, std::vector<int> rowIdx, std::vector<int> colIdx) : dim(rowIdx.size() - 1),
                                                                                                           values(std::move(values)),
                                                                                                           rowIdx(std::move(rowIdx)),
                                                                                                           colIdx(std::move(colIdx)) {}
// SparseMatrix::SparseMatrix(const SparseMatrix& other) : dim(other.dim),
//                                                         values(other.values),
//                                                         rowIdx(other.rowIdx),
//                                                         colIdx(other.colIdx) {}

SparseMatrix::SparseMatrix(SparseMatrix&& other) : dim(std::move(other.dim)),
                                                   values(std::move(other.values)),
                                                   rowIdx(std::move(other.rowIdx)),
                                                   colIdx(std::move(other.colIdx)) {}

// SparseMatrix& SparseMatrix::operator=(const SparseMatrix& other) {
//     this->dim = other.dim;
//     this->values = other.values;
//     this->rowIdx = other.rowIdx;
//     this->colIdx = other.colIdx;
//     return *this;
// }

SparseMatrix& SparseMatrix::operator=(SparseMatrix&& other) {
    this->dim = std::move(other.dim);
    this->values = std::move(other.values);
    this->rowIdx = std::move(other.rowIdx);
    this->colIdx = std::move(other.colIdx);
    return *this;
}

SparseMatrix SparseMatrix::fromFile(std::string &otherFileName) {
    std::ifstream otherFile(otherFileName);

    int width, height, nonZerosCount, nonZerosPerRow;

    otherFile >> width >> height >> nonZerosCount >> nonZerosPerRow;
    assert(width == height);

    std::vector<double> nonZeros(nonZerosCount);
    for (uint i = 0; i < nonZeros.size(); i++) {
        otherFile >> nonZeros[i];
    }

    std::vector<int> rowIdx(width + 1);
    for (uint i = 0; i < rowIdx.size(); i++) {
        otherFile >> rowIdx[i];
    }

    std::vector<int> colIdx(nonZerosCount);
    for (uint i = 0; i < colIdx.size(); i++) {
        otherFile >> colIdx[i];
    }

    otherFile.close();
    return SparseMatrix(nonZeros, rowIdx, colIdx);
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

template <>
void communication::Send<SparseMatrix>(SparseMatrix& mat, int destProcessId, int tag, MPI_Comm comm) {
    PackedData data = std::move(mat.pack());
    communication::Send<PackedData>(data, destProcessId, tag, comm);
}

template <>
communication::Request communication::Isend<SparseMatrix>(SparseMatrix& mat, int destProcessId, int tag, MPI_Comm comm) {
    PackedData data = mat.pack();
    return communication::Isend<PackedData>(data, destProcessId, tag, comm);
}

template <>
void communication::Recv<SparseMatrix>(SparseMatrix& mat, int srcProcessId, int tag, MPI_Comm comm) {
    PackedData data = communication::Recv<PackedData>(srcProcessId, tag, comm);
    mat = std::move(SparseMatrix::unpack(data));
}

template <>
SparseMatrix communication::Recv<SparseMatrix>(int srcProcessId, int tag, MPI_Comm comm) {
    SparseMatrix mat;
    communication::Recv<SparseMatrix>(mat, srcProcessId, tag, comm);
    return mat;
}

MatrixFragment SparseMatrix::fragmentOfProcess(Environment& env, int processId) {
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

void SparseMatrix::print() {
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

void SparseMatrix::printShort() {
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

void SparseMatrix::Iterator::adjustRowIdx() {
    while (curRowIdx < (int)other->rowIdx.size()
        && !(other->rowIdx[curRowIdx] <= curValueIdx
            && curValueIdx < other->rowIdx[curRowIdx + 1])) {
        curRowIdx++;
    }
}

SparseMatrix::Iterator::Iterator(SparseMatrix* other, int curValueIdx) noexcept : other(other),
                                                            curRowIdx(0),
                                                            curValueIdx(curValueIdx) {
    adjustRowIdx();
}

SparseMatrix::Field SparseMatrix::Iterator::operator*() const {
    return {
        {
            this->curRowIdx,
            this->other->colIdx[this->curValueIdx]
        },
        this->other->values[this->curValueIdx]
    };
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
    return a.other == b.other
        && (a.curValueIdx == b.curValueIdx
            || (a.curValueIdx >= (int)a.other->values.size()
                && b.curValueIdx >= (int)b.other->values.size()));
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

DenseMatrix::DenseMatrix() {}

// DenseMatrix::DenseMatrix(const DenseMatrix& other) : dim(other.dim),
//                                                      numColumns(other.numColumns),
//                                                      values(other.values) {}

DenseMatrix::DenseMatrix(DenseMatrix&& other) : dim(std::move(other.dim)),
                                                numColumns(std::move(other.numColumns)),
                                                values(std::move(other.values)) {}

// DenseMatrix& DenseMatrix::operator=(const DenseMatrix& other) {
//     this->dim = other.dim;
//     this->numColumns = other.numColumns;
//     this->values = other.values;
//     return *this;
// }

DenseMatrix& DenseMatrix::operator=(DenseMatrix&& other) {
    this->dim = std::move(other.dim);
    this->numColumns = std::move(other.numColumns);
    this->values = std::move(other.values);
    return *this;
}

DenseMatrix DenseMatrix::blank(int numRows, int numColumns) {
    std::vector<DenseMatrix::RowType> values(numRows, RowType(numColumns, 0));
    return DenseMatrix(numRows, numColumns, values);
}

DenseMatrix::DenseMatrix(int dim, int numColumns, std::vector<DenseMatrix::RowType>& values) : dim(dim),
                                                                                               numColumns(numColumns),
                                                                                               values(std::move(values)) {}

DenseMatrix DenseMatrix::generate(MatrixFragment& frag, int seed) {
    MatrixIndex start, end;
    std::tie(start, end) = frag;
    int numColumns = end.col - start.col + 1;
    int numRows = end.row - start.row + 1;
    assert(numColumns > 0);
    assert(numRows > 0);

    std::vector<DenseMatrix::RowType> values(numRows, DenseMatrix::RowType(numColumns));
    for (int r = start.row; r <= end.row; r++) {
        for (int c = start.col; c <= end.col; c++) {
            values[r - start.row][c - start.col] = generate_double(seed, r, c);
        }
    }

    return DenseMatrix(numRows, numColumns, values);
}


PackedData DenseMatrix::pack(MPI_Comm comm) {
    PackedData buf;
    int size, pos = 0, packSize;

    // pack number of columns
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = this->numColumns;
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    // pack number of values
    MPI_Pack_size(1, MPI_INT, comm, &size);
    buf.resize(pos + size);
    packSize = this->dim;
    MPI_Pack(&packSize, 1, MPI_INT, buf.data(), buf.size(), &pos, comm);

    // pack each row
    for (auto &r: values) {
        MPI_Pack_size(this->numColumns, MPI_DOUBLE, comm, &size);
        buf.resize(pos + size);
        MPI_Pack(r.data(), r.size(), MPI_DOUBLE, buf.data(), buf.size(), &pos, comm);
    }

    return buf;
}

DenseMatrix DenseMatrix::unpack(PackedData& buf, MPI_Comm comm) {
    int pos = 0;

    // unpack number of columns
    int numColumns;
    MPI_Unpack(buf.data(), buf.size(), &pos, &numColumns, 1, MPI_INT, comm);

    // unpack number of values
    int dim;
    MPI_Unpack(buf.data(), buf.size(), &pos, &dim, 1, MPI_INT, comm);

    // unpack each row
    std::vector<DenseMatrix::RowType> values(dim, DenseMatrix::RowType(numColumns));
    for (auto &r: values) {
        MPI_Unpack(buf.data(), buf.size(), &pos, r.data(), numColumns, MPI_DOUBLE, comm);
    }

    return DenseMatrix(dim, numColumns, values);
}

template <>
void communication::Send<DenseMatrix>(DenseMatrix& mat, int destProcessId, int tag, MPI_Comm comm) {
    PackedData data = std::move(mat.pack());
    communication::Send<PackedData>(data, destProcessId, tag, comm);
}

template <>
communication::Request communication::Isend<DenseMatrix>(DenseMatrix& mat, int destProcessId, int tag, MPI_Comm comm) {
    PackedData data = mat.pack();
    return communication::Isend<PackedData>(data, destProcessId, tag, comm);
}

template <>
void communication::Recv<DenseMatrix>(DenseMatrix& mat, int srcProcessId, int tag, MPI_Comm comm) {
    PackedData data;
    communication::Recv<PackedData>(data, srcProcessId, tag, comm);

    mat = std::move(DenseMatrix::unpack(data));
}

template <>
DenseMatrix communication::Recv<DenseMatrix>(int srcProcessId, int tag, MPI_Comm comm) {
    DenseMatrix mat;
    communication::Recv<DenseMatrix>(mat, srcProcessId, tag, comm);
    return mat;
}

int getDenseMatrixFirstColumn(Environment& env, int processId) {
    int baseColumnsPerProcess = env.matrixDimension / env.numProcesses;
    return baseColumnsPerProcess * processId
         + std::min(processId, env.matrixDimension % env.numProcesses);
}

MatrixFragment DenseMatrix::fragmentOfProcess(Environment& env, int processId) {
    int colBeginIncl = getDenseMatrixFirstColumn(env, processId);
    int colEndExcl = getDenseMatrixFirstColumn(env, processId + 1); 

    return {
        { 0, colBeginIncl },
        { env.matrixDimension - 1, colEndExcl - 1 }
    };
}

void DenseMatrix::print() {
    for (auto &row: this->values) {
        for (auto &field: row) {
            std::cout << std::setprecision(5) << std::fixed << "    " << field;
        }
        std::cout << std::endl;
    }
}