#ifndef NNUE_ARCHITECTURES_CUH
#define NNUE_ARCHITECTURES_CUH

// This file contains only the struct definitions for the NNUE architecture.
// It has no other dependencies and can be safely included by multiple files.

struct BucketEmbedding {
    // These will point to __device__ memory
    float* W_white_piece;
    float* W_black_piece;
    float* W_white_castle;
    float* W_black_castle;
    float* W_white_ep;
    float* W_black_ep;
    float* W_white_fifty;
    float* W_black_fifty;
};

struct ResidualBlock {
    // These will point to __device__ memory
    float *fc1_w, *fc2_w;
    float *fc1_b, *fc2_b, *ln1_w, *ln1_b, *ln2_w, *ln2_b;
};

struct MLPHead {
    // These will point to __device__ memory
    float *fc1_w, *fc2_w, *fc_out_w;
    float *fc1_b, *ln1_w, *ln1_b, *fc2_b, *ln2_w, *ln2_b, *fc_out_b;
    ResidualBlock blocks[12];
};

#endif // NNUE_ARCHITECTURES_CUH
