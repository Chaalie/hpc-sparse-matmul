#include "common.h"

int main(int argc, char* argv[]) {
    InputOptions::parse(argc, argv).print();

    return 0;
}