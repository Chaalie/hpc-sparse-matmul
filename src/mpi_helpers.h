#ifndef __PACKED_H__
#define __PACKED_H__

#include <mpi.h>
#include <vector>

typedef std::vector<char> PackedData;

template <typename T>
T unpack(PackedData& packedData, MPI_Comm comm) {
    return unpack<T>(packedData.data(), packedData.size(), comm);
};

template <typename T>
T unpack(char *buf, int size, MPI_Comm comm);

template <typename T>
PackedData pack(T& data, MPI_Comm comm);

#endif /* __PACKED_H__ */