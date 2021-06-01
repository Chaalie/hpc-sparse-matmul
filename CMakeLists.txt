cmake_minimum_required(VERSION 3.5)
project (MIMUW-matgen)

SET(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++14  -g3 -Wall -O0")

add_executable(matrix-mul
    src/generator/densematgen.cpp
    src/generator/densematgen.h
    src/common.h
    src/utils.h
    src/utils.cpp
    src/mpi_helpers.h
    src/program_options.h
    src/program_options.cpp
    src/replication_group.h
    src/replication_group.cpp
    src/multiplication.h
    src/multiplication.cpp
    src/matrix.h
    src/matrix.cpp
    src/main.cpp)

target_link_libraries(matrix-mul ${MPI_C_LIBRARIES})