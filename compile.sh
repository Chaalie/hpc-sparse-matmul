#!/bin/bash

rm -rf build; mkdir build; cd build; CXX=mpic++ CC=mpicc cmake ..; make