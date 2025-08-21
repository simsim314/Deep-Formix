#ifndef EMBEDDING_NNUE_CUH
#define EMBEDDING_NNUE_CUH

#include <cstdint>
#include "chess.h"
#include "cuda_adapters.cuh"
#include "nnue_architectures.cuh"

namespace NNUE_EMBEDDING {

// One-time initialization of the embedding tables.
void init(const float* model_data);

// Releases all GPU memory allocated by the embedding tables.
void cleanup();

// Computes the 2240-feature input vector for the NNUE.
__device__ void generate_embedding(float* x_emb, const HexaBitBoardPosition* board, int bucket_idx);

// Determines the correct bucket index for a given board position.
__device__ int get_bucket_index(const HexaBitBoardPosition* board);

// Generates the 64-bit key used for bucket lookup.
__device__ uint64_t get_map_key_device(const HexaBitBoardPosition* board);

} // namespace NNUE_EMBEDDING

#endif // EMBEDDING_NNUE_CUH
