// src/main.cu
#include "uci.cuh"
#include "MoveGeneratorBitboard.h"

int main(int argc, char** argv) {
    MoveGeneratorBitboard::init();
    UCI::loop(argc, argv);
    return 0;
}
