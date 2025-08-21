#ifndef EVALUATE_NNUE_CUH
#define EVALUATE_NNUE_CUH

#include "chess.h"
#include "nnue_architectures.cuh"

namespace NNUE_EVALUATE {

// Workspace used by evaluate paths (256 + 32 floats)
constexpr int NNUE_FORWARD_WORKSPACE_SIZE = 256 + 32;

// One-time initialization of the NNUE model (+ fc1 precompute projections).
void init(const float* model_data);

// Releases all GPU memory allocated by the NNUE.
void cleanup();

// Legacy entry (expects x_emb already built). Kept for compatibility.
__device__ int evaluate_embedding(const float* x_emb, int bucket_idx);

// New fast path: evaluate using precomputed fc1 projections directly from the board.
__device__ int evaluate_prebaked(const HexaBitBoardPosition* position);

// ── Public device symbols used by evaluate.cu ────────────────────────────────

// Precomputed fc1 projections (per bucket, per side-flag)
struct Fc1Precomp {
    float* piece;   // [64*12*256]
    float* castle;  // [4*256]
    float* ep;      // [8*256]
    float* fifty;   // [2*256]
};

// Device globals defined in evaluate_nnue.cu
extern __device__ Fc1Precomp d_fc1_precomp[32][2];
extern __device__ MLPHead    d_networks[32];
extern __device__ float*     d_bins;

// Helper device functions (implemented in evaluate_nnue.cu)
__device__ void apply_gelu(float* data, int n);
__device__ void layer_norm(float* x, const float* weight, const float* bias, int N);
__device__ void mat_vec_mul_add_bias(float* out, const float* mat, const float* vec,
                                     const float* bias, int rows, int cols);
__device__ void residual_block_forward(float* x, const ResidualBlock& block);
// Returns cp directly; full forward stays inside NNUE
__device__ int evaluate_cp(const HexaBitBoardPosition* position);

} // namespace NNUE_EVALUATE

#endif // EVALUATE_NNUE_CUH
