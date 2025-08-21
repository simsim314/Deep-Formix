// src/evaluate.cu
#include "evaluate.cuh"
#include "embedding_nnue.cuh"
#include "evaluate_nnue.cuh"
#include "switches.h" 
#include "uci.cuh" // For access to global network path

#include <vector>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>

namespace EVALUATION {

std::vector<float> h_model_weights;

void init_nnue() {
    // Use the global variable for the model path
    std::ifstream model_file(UCI::g_nnue_model_path, std::ios::binary | std::ios::ate);
    
    if (!model_file) {
        std::fprintf(stderr, "FATAL: Could not open model file: %s\n", UCI::g_nnue_model_path.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::streamsize size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);

    size_t num_floats = static_cast<size_t>(size) / sizeof(float);
    h_model_weights.resize(num_floats);

    if (!model_file.read(reinterpret_cast<char*>(h_model_weights.data()), size)) {
        std::fprintf(stderr, "FATAL: Failed to read model file: %s\n", UCI::g_nnue_model_path.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Model file read successfully. Total floats: " << num_floats << std::endl;

    // Initialize bucket mapping and NNUE network (forward fully inside NNUE_EVALUATE)
    NNUE_EMBEDDING::init(h_model_weights.data());
    NNUE_EVALUATE::init(h_model_weights.data());
}

void cleanup_nnue() {
    NNUE_EMBEDDING::cleanup();
    NNUE_EVALUATE::cleanup();
    h_model_weights.clear();
}

// Legacy (unused)
__device__ int evaluatePosition(HexaBitBoardPosition* /*position*/) {
    return 0;
}

// Thin wrapper: get cp directly from NNUE (all forward logic stays in evaluate_nnue)
__device__ int evaluatePosition_nnue(const HexaBitBoardPosition* position) {
    // Call the workspace-free version of the evaluation function.
    return NNUE_EVALUATE::evaluate_cp(position);
}

}  // namespace EVALUATION
