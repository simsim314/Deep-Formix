#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include "macros.cuh"
#include "chess.h"

namespace EVALUATION {

// Legacy placeholder (kept for compatibility)
__device__ int evaluatePosition(HexaBitBoardPosition* position);

// One-time initialization for the NNUE model (loads model, embeddings map, and NNUE weights).
void init_nnue();

// Releases all GPU memory allocated by the NNUE.
void cleanup_nnue();

/**
 * Thin wrapper that returns a centipawn score from side-to-move’s perspective.
 * The entire network forward pass happens inside NNUE_EVALUATE; this function
 * only delegates to it and requires NO per-thread workspace allocation from the caller.
 *
 * position  - device pointer to board state
 */
__device__ int evaluatePosition_nnue(const HexaBitBoardPosition* position);

} // namespace EVALUATION

#endif // EVALUATE_H_INCLUDED
