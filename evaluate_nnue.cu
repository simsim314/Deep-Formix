#include "evaluate_nnue.cuh"
#include "embedding_nnue.cuh"     // get_bucket_index()
#include "bitboard_constants.h"
#include "bitboard_utils.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <math_constants.h>

namespace NNUE_EVALUATE {

// ============================================================================
// DEVICE GLOBALS
// ============================================================================
__device__ MLPHead d_networks[32];
__device__ float*  d_bins = nullptr;  // set via init()
__device__ Fc1Precomp d_fc1_precomp[32][2];

// Host tracking for cleanup
static Fc1Precomp h_fc1_precomp[32][2];

// ============================================================================
// DEVICE HELPERS
// ============================================================================
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / CUDART_PI_F) * (x + 0.044715f * x * x * x)));
}
__device__ void apply_gelu(float* data, int n) {
    for (int i = 0; i < n; ++i) data[i] = gelu(data[i]);
}
__device__ void layer_norm(float* x, const float* weight, const float* bias, int N) {
    float mean = 0.0f;
    for (int i = 0; i < N; ++i) mean += x[i];
    mean /= N;

    float sq_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = x[i] - mean;
        sq_sum += d * d;
    }
    float inv_std = rsqrtf(sq_sum / N + 1e-5f);
    for (int i = 0; i < N; ++i) {
        x[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
    }
}
__device__ void mat_vec_mul_add_bias(float* out, const float* mat, const float* vec,
                                     const float* bias, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        const float* row = mat + i * cols;
        for (int j = 0; j < cols; ++j) sum += row[j] * vec[j];
        out[i] = sum + bias[i];
    }
}
__device__ void residual_block_forward(float* x, const ResidualBlock& block) {
    float y[32];
    for (int i = 0; i < 32; ++i) y[i] = x[i];

    float tmp[32];
    mat_vec_mul_add_bias(tmp, block.fc1_w, x, block.fc1_b, 32, 32);
    apply_gelu(tmp, 32);
    layer_norm(tmp, block.ln1_w, block.ln1_b, 32);

    mat_vec_mul_add_bias(x, block.fc2_w, tmp, block.fc2_b, 32, 32);
    apply_gelu(x, 32);
    layer_norm(x, block.ln2_w, block.ln2_b, 32);

    for (int i = 0; i < 32; ++i) x[i] += y[i];
}

// ============================================================================
// DEVICE FORWARD – NEW FAST PATH
// ============================================================================
__device__ int evaluate_prebaked(const HexaBitBoardPosition* position) {
    if (position->kings == 0) return 0;

    // Pick bucket based on board
    int bucket_idx = NNUE_EMBEDDING::get_bucket_index(position);
    const MLPHead& net = d_networks[bucket_idx];

    // side flag: 0 = white to move → use BLACK embeddings (inverted),
    //            1 = black to move → use WHITE embeddings
    int sflag = (position->chance == WHITE) ? 0 : 1;
    const Fc1Precomp& pc = d_fc1_precomp[bucket_idx][sflag];

    // These are now local, per-thread arrays, not pointers into a global workspace.
    float x1[256];
    float x2[32];

    // 1) Build x1 from precomputed projections
    for (int r = 0; r < 256; ++r) x1[r] = 0.0f;

    uint64 all_pawns  = position->pawns & RANKS2TO7;
    uint64 all_pieces = position->kings | all_pawns | position->knights | position->bishopQueens | position->rookQueens;
    uint64 queens     = position->bishopQueens & position->rookQueens;

    uint64 tmp = all_pieces;
    while (tmp) {
        uint64 bb = getOne(tmp);
        int sq = bitScan(bb);

        int piece_type_idx = -1;
        if      (bb & all_pawns)                                piece_type_idx = 0;
        else if (bb & position->knights)                        piece_type_idx = 1;
        else if (bb & (position->bishopQueens & ~queens))       piece_type_idx = 2;
        else if (bb & (position->rookQueens   & ~queens))       piece_type_idx = 3;
        else if (bb & queens)                                   piece_type_idx = 4;
        else if (bb & position->kings)                          piece_type_idx = 5;

        int color_off = (bb & position->whitePieces) ? 0 : 6;
        int pidx = piece_type_idx + color_off;

        const float* v = pc.piece + (((sq * 12) + pidx) * 256);
        for (int r = 0; r < 256; ++r) x1[r] += v[r];

        tmp ^= bb;
    }

    if (position->whiteCastle & CASTLE_FLAG_KING_SIDE)  { const float* v = pc.castle + 0*256; for (int r=0;r<256;++r) x1[r] += v[r]; }
    if (position->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) { const float* v = pc.castle + 1*256; for (int r=0;r<256;++r) x1[r] += v[r]; }
    if (position->blackCastle & CASTLE_FLAG_KING_SIDE)  { const float* v = pc.castle + 2*256; for (int r=0;r<256;++r) x1[r] += v[r]; }
    if (position->blackCastle & CASTLE_FLAG_QUEEN_SIDE) { const float* v = pc.castle + 3*256; for (int r=0;r<256;++r) x1[r] += v[r]; }

    if (position->enPassent != 0) {
        int f = position->enPassent - 1;
        const float* v = pc.ep + f * 256;
        for (int r = 0; r < 256; ++r) x1[r] += v[r];
    }

    float a = fminf((float)position->halfMoveCounter / 100.0f, 1.0f);
    const float* f0 = pc.fifty + 0*256;
    const float* f1 = pc.fifty + 1*256;
    for (int r = 0; r < 256; ++r) x1[r] += (1.0f - a) * f0[r] + a * f1[r];

    // 2) Add fc1 bias → GELU → LN
    for (int r = 0; r < 256; ++r) x1[r] += net.fc1_b[r];
    apply_gelu(x1, 256);
    layer_norm(x1, net.ln1_w, net.ln1_b, 256);

    // 3) fc2 → residual stack → head
    mat_vec_mul_add_bias(x2, net.fc2_w, x1, net.fc2_b, 32, 256);
    apply_gelu(x2, 32);
    layer_norm(x2, net.ln2_w, net.ln2_b, 32);

    for (int i = 0; i < 12; ++i) residual_block_forward(x2, net.blocks[i]);
    apply_gelu(x2, 32);

    float logits[51];
    mat_vec_mul_add_bias(logits, net.fc_out_w, x2, net.fc_out_b, 51, 32);

    float mx = -1e30f;
    for (int i = 0; i < 51; ++i) if (logits[i] > mx) mx = logits[i];

    float exp_sum = 0.0f;
    for (int i = 0; i < 51; ++i) {
        logits[i] = expf(logits[i] - mx);
        exp_sum += logits[i];
    }

    float p_win = 0.0f;
    for (int i = 0; i < 51; ++i) p_win += (logits[i] / exp_sum) * d_bins[i];

    float score_cp = (p_win > 0.999f) ? 10000.0f : (p_win < 0.001f) ? -10000.0f : 290.68f * tanf(3.096f * (p_win - 0.5f));
    int score_stm = (int)score_cp;
    return (position->chance == WHITE) ? score_stm : -score_stm;
}

// Keep legacy entry for compatibility if anything still calls it
__device__ int evaluate_embedding(const float* x_emb, int bucket_idx) {
    const MLPHead& net = d_networks[bucket_idx];

    float x1[256];
    float x2[32];

    // x1 = fc1_w * x_emb + fc1_b
    mat_vec_mul_add_bias(x1, net.fc1_w, x_emb, net.fc1_b, 256, 2240);
    apply_gelu(x1, 256);
    layer_norm(x1, net.ln1_w, net.ln1_b, 256);

    mat_vec_mul_add_bias(x2, net.fc2_w, x1, net.fc2_b, 32, 256);
    apply_gelu(x2, 32);
    layer_norm(x2, net.ln2_w, net.ln2_b, 32);

    for (int i = 0; i < 12; ++i) residual_block_forward(x2, net.blocks[i]);
    apply_gelu(x2, 32);

    float logits[51];
    mat_vec_mul_add_bias(logits, net.fc_out_w, x2, net.fc_out_b, 51, 32);

    float mx = -1e30f;
    for (int i = 0; i < 51; ++i) if (logits[i] > mx) mx = logits[i];

    float exp_sum = 0.0f;
    for (int i = 0; i < 51; ++i) {
        logits[i] = expf(logits[i] - mx);
        exp_sum += logits[i];
    }

    float p_win = 0.0f;
    for (int i = 0; i < 51; ++i) p_win += (logits[i] / exp_sum) * d_bins[i];

    float score_cp = (p_win > 0.999f) ? 10000.0f : (p_win < 0.001f) ? -10000.0f : 290.68f * tanf(3.096f * (p_win - 0.5f));
    return (int)score_cp;
}

// ============================================================================
// HOST INIT: load model, upload weights and precompute fc1 projections
// ============================================================================
void init(const float* model_data) {
    // Upload bins
    std::vector<float> h_bins(51);
    for (int i = 0; i < 51; ++i) h_bins[i] = (float)i / 50.0f;
    float* d_bins_tmp = nullptr;
    cudaMalloc(&d_bins_tmp, 51 * sizeof(float));
    cudaMemcpy(d_bins_tmp, h_bins.data(), 51 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_bins, &d_bins_tmp, sizeof(d_bins_tmp));

    // Layout constants (floats) – matches your file format
    const size_t EMB_WHITE_PIECE  = 64 * 12 * 32;
    const size_t EMB_BLACK_PIECE  = 64 * 12 * 32;
    const size_t EMB_WHITE_CASTLE = 4  * 32;
    const size_t EMB_BLACK_CASTLE = 4  * 32;
    const size_t EMB_WHITE_EP     = 8  * 32;
    const size_t EMB_BLACK_EP     = 8  * 32;
    const size_t EMB_WHITE_FIFTY  = 2  * 32;
    const size_t EMB_BLACK_FIFTY  = 2  * 32;
    const size_t EMB_SIZE =
        EMB_WHITE_PIECE + EMB_BLACK_PIECE +
        EMB_WHITE_CASTLE + EMB_BLACK_CASTLE +
        EMB_WHITE_EP + EMB_BLACK_EP +
        EMB_WHITE_FIFTY + EMB_BLACK_FIFTY;

    const size_t NET_FC1_W   = 256 * 2240;
    const size_t NET_FC1_B   = 256;
    const size_t NET_LN1_W   = 256;
    const size_t NET_LN1_B   = 256;
    const size_t NET_FC2_W   = 32  * 256;
    const size_t NET_FC2_B   = 32;
    const size_t NET_LN2_W   = 32;
    const size_t NET_LN2_B   = 32;
    const size_t RBLOCK      = (32*32) + 32 + 32 + 32 + (32*32) + 32 + 32 + 32;
    const size_t NET_BLOCKS  = 12 * RBLOCK;
    const size_t NET_FCOUT_W = 51 * 32;
    const size_t NET_FCOUT_B = 51;
    const size_t NET_SIZE    = NET_FC1_W + NET_FC1_B + NET_LN1_W + NET_LN1_B
                             + NET_FC2_W + NET_FC2_B + NET_LN2_W + NET_LN2_B
                             + NET_BLOCKS + NET_FCOUT_W + NET_FCOUT_B;

    // Upload networks and precompute vectors
    for (int b = 0; b < 32; ++b) {
        // Upload this bucket's network weights
        const float* p = model_data + b * (EMB_SIZE + NET_SIZE) + EMB_SIZE;

        MLPHead net;
        cudaMalloc(&net.fc1_w, sizeof(float) * NET_FC1_W); cudaMemcpy(net.fc1_w, p, sizeof(float) * NET_FC1_W, cudaMemcpyHostToDevice); p += NET_FC1_W;
        cudaMalloc(&net.fc1_b, sizeof(float) * NET_FC1_B); cudaMemcpy(net.fc1_b, p, sizeof(float) * NET_FC1_B, cudaMemcpyHostToDevice); p += NET_FC1_B;
        cudaMalloc(&net.ln1_w, sizeof(float) * NET_LN1_W); cudaMemcpy(net.ln1_w, p, sizeof(float) * NET_LN1_W, cudaMemcpyHostToDevice); p += NET_LN1_W;
        cudaMalloc(&net.ln1_b, sizeof(float) * NET_LN1_B); cudaMemcpy(net.ln1_b, p, sizeof(float) * NET_LN1_B, cudaMemcpyHostToDevice); p += NET_LN1_B;
        cudaMalloc(&net.fc2_w, sizeof(float) * NET_FC2_W); cudaMemcpy(net.fc2_w, p, sizeof(float) * NET_FC2_W, cudaMemcpyHostToDevice); p += NET_FC2_W;
        cudaMalloc(&net.fc2_b, sizeof(float) * NET_FC2_B); cudaMemcpy(net.fc2_b, p, sizeof(float) * NET_FC2_B, cudaMemcpyHostToDevice); p += NET_FC2_B;
        cudaMalloc(&net.ln2_w, sizeof(float) * NET_LN2_W); cudaMemcpy(net.ln2_w, p, sizeof(float) * NET_LN2_W, cudaMemcpyHostToDevice); p += NET_LN2_W;
        cudaMalloc(&net.ln2_b, sizeof(float) * NET_LN2_B); cudaMemcpy(net.ln2_b, p, sizeof(float) * NET_LN2_B, cudaMemcpyHostToDevice); p += NET_LN2_B;
        for (int i = 0; i < 12; ++i) {
            cudaMalloc(&net.blocks[i].fc1_w, sizeof(float) * 32 * 32); cudaMemcpy(net.blocks[i].fc1_w, p, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice); p += 32 * 32;
            cudaMalloc(&net.blocks[i].fc1_b, sizeof(float) * 32);      cudaMemcpy(net.blocks[i].fc1_b, p, sizeof(float) * 32,      cudaMemcpyHostToDevice); p += 32;
            cudaMalloc(&net.blocks[i].ln1_w, sizeof(float) * 32);      cudaMemcpy(net.blocks[i].ln1_w, p, sizeof(float) * 32,      cudaMemcpyHostToDevice); p += 32;
            cudaMalloc(&net.blocks[i].ln1_b, sizeof(float) * 32);      cudaMemcpy(net.blocks[i].ln1_b, p, sizeof(float) * 32,      cudaMemcpyHostToDevice); p += 32;
            cudaMalloc(&net.blocks[i].fc2_w, sizeof(float) * 32 * 32); cudaMemcpy(net.blocks[i].fc2_w, p, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice); p += 32 * 32;
            cudaMalloc(&net.blocks[i].fc2_b, sizeof(float) * 32);      cudaMemcpy(net.blocks[i].fc2_b, p, sizeof(float) * 32,      cudaMemcpyHostToDevice); p += 32;
            cudaMalloc(&net.blocks[i].ln2_w, sizeof(float) * 32);      cudaMemcpy(net.blocks[i].ln2_w, p, sizeof(float) * 32,      cudaMemcpyHostToDevice); p += 32;
            cudaMalloc(&net.blocks[i].ln2_b, sizeof(float) * 32);      cudaMemcpy(net.blocks[i].ln2_b, p, sizeof(float) * 32,      cudaMemcpyHostToDevice); p += 32;
        }
        cudaMalloc(&net.fc_out_w, sizeof(float) * NET_FCOUT_W); cudaMemcpy(net.fc_out_w, p, sizeof(float) * NET_FCOUT_W, cudaMemcpyHostToDevice); p += NET_FCOUT_W;
        cudaMalloc(&net.fc_out_b, sizeof(float) * NET_FCOUT_B); cudaMemcpy(net.fc_out_b, p, sizeof(float) * NET_FCOUT_B, cudaMemcpyHostToDevice); p += NET_FCOUT_B;

        cudaMemcpyToSymbol(d_networks, &net, sizeof(MLPHead),
                           b * sizeof(MLPHead), cudaMemcpyHostToDevice);

        // Host pointers for this bucket’s embeddings
        const float* bucket_base   = model_data + b * (EMB_SIZE + NET_SIZE);
        const float* W_white_piece = bucket_base;
        const float* W_black_piece = W_white_piece + EMB_WHITE_PIECE;
        const float* W_white_castle= W_black_piece + EMB_BLACK_PIECE;
        const float* W_black_castle= W_white_castle + EMB_WHITE_CASTLE;
        const float* W_white_ep    = W_black_castle + EMB_BLACK_CASTLE;
        const float* W_black_ep    = W_white_ep + EMB_WHITE_EP;
        const float* W_white_fifty = W_black_ep + EMB_BLACK_EP;
        const float* W_black_fifty = W_white_fifty + EMB_WHITE_FIFTY;

        // Host pointer to fc1 weights for this bucket (row-major 256×2240)
        const float* fc1_w_host = model_data + b * (EMB_SIZE + NET_SIZE) + EMB_SIZE;

        auto mul_block_256x32 = [&](int col_base, const float* F32, float* out256) {
            for (int r = 0; r < 256; ++r) {
                const float* row = fc1_w_host + r * 2240 + col_base;
                float sum = 0.0f;
                for (int k = 0; k < 32; ++k) sum += row[k] * F32[k];
                out256[r] = sum;
            }
        };

        for (int side = 0; side < 2; ++side) {
            const float* piece_tab  = (side == 0) ? W_black_piece  : W_white_piece;
            const float* castle_tab = (side == 0) ? W_black_castle : W_white_castle;
            const float* ep_tab     = (side == 0) ? W_black_ep     : W_white_ep;
            const float* fifty_tab  = (side == 0) ? W_black_fifty  : W_white_fifty;

            const size_t PIECE_SIZE  = 64 * 12 * 256;
            const size_t CASTLE_SIZE = 4  * 256;
            const size_t EP_SIZE     = 8  * 256;
            const size_t FIFTY_SIZE  = 2  * 256;

            std::vector<float> h_piece_proj(PIECE_SIZE);
            std::vector<float> h_castle_proj(CASTLE_SIZE);
            std::vector<float> h_ep_proj(EP_SIZE);
            std::vector<float> h_fifty_proj(FIFTY_SIZE);

            for (int sq = 0; sq < 64; ++sq) {
                const int col_base = sq * 32;
                for (int pidx = 0; pidx < 12; ++pidx) {
                    const float* F32 = piece_tab + (sq * 12 + pidx) * 32;
                    float* out256    = &h_piece_proj[((sq * 12 + pidx) * 256)];
                    mul_block_256x32(col_base, F32, out256);
                }
            }
            for (int i = 0; i < 4; ++i) {
                const float* F32 = castle_tab + i * 32;
                float* out256    = &h_castle_proj[i * 256];
                mul_block_256x32(2048 + i * 32, F32, out256);
            }
            for (int f = 0; f < 8; ++f) {
                const float* F32 = ep_tab + f * 32;
                float* out256    = &h_ep_proj[f * 256];
                mul_block_256x32(2176, F32, out256);
            }
            for (int j = 0; j < 2; ++j) {
                const float* F32 = fifty_tab + j * 32;
                float* out256    = &h_fifty_proj[j * 256];
                mul_block_256x32(2208, F32, out256);
            }

            float *d_piece=nullptr, *d_castle=nullptr, *d_ep=nullptr, *d_fifty=nullptr;
            cudaMalloc(&d_piece,  PIECE_SIZE  * sizeof(float)); cudaMemcpy(d_piece,  h_piece_proj.data(),  PIECE_SIZE  * sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc(&d_castle, CASTLE_SIZE * sizeof(float)); cudaMemcpy(d_castle, h_castle_proj.data(), CASTLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc(&d_ep,     EP_SIZE     * sizeof(float)); cudaMemcpy(d_ep,     h_ep_proj.data(),     EP_SIZE     * sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc(&d_fifty,  FIFTY_SIZE  * sizeof(float)); cudaMemcpy(d_fifty,  h_fifty_proj.data(),  FIFTY_SIZE  * sizeof(float), cudaMemcpyHostToDevice);

            h_fc1_precomp[b][side] = { d_piece, d_castle, d_ep, d_fifty };
            size_t ofs = (b * 2 + side) * sizeof(Fc1Precomp);
            cudaMemcpyToSymbol(d_fc1_precomp, &h_fc1_precomp[b][side], sizeof(Fc1Precomp), ofs, cudaMemcpyHostToDevice);
        }
    }

    std::cout << "NNUE network weights loaded to GPU (prebaked fc1 projections enabled)." << std::endl;
}

// Thin cp API for external callers (keeps all forward logic inside this TU)
__device__ int evaluate_cp(const HexaBitBoardPosition* position) {
    return evaluate_prebaked(position);
}

void cleanup() {
    // Release bins
    float* d_bins_tmp = nullptr;
    cudaMemcpyFromSymbol(&d_bins_tmp, d_bins, sizeof(d_bins_tmp));
    if (d_bins_tmp) cudaFree(d_bins_tmp);
    d_bins_tmp = nullptr;
    cudaMemcpyToSymbol(d_bins, &d_bins_tmp, sizeof(d_bins_tmp));

    // Release precomputed projections
    for (int b = 0; b < 32; ++b) {
        for (int s = 0; s < 2; ++s) {
            if (h_fc1_precomp[b][s].piece)  cudaFree(h_fc1_precomp[b][s].piece),  h_fc1_precomp[b][s].piece  = nullptr;
            if (h_fc1_precomp[b][s].castle) cudaFree(h_fc1_precomp[b][s].castle), h_fc1_precomp[b][s].castle = nullptr;
            if (h_fc1_precomp[b][s].ep)     cudaFree(h_fc1_precomp[b][s].ep),     h_fc1_precomp[b][s].ep     = nullptr;
            if (h_fc1_precomp[b][s].fifty)  cudaFree(h_fc1_precomp[b][s].fifty),  h_fc1_precomp[b][s].fifty  = nullptr;
        }
    }

    // Release networks
    MLPHead nets[32];
    cudaMemcpyFromSymbol(nets, d_networks, sizeof(nets));
    for (int i = 0; i < 32; ++i) {
        if (nets[i].fc1_w) cudaFree(nets[i].fc1_w);
        if (nets[i].fc1_b) cudaFree(nets[i].fc1_b);
        if (nets[i].ln1_w) cudaFree(nets[i].ln1_w);
        if (nets[i].ln1_b) cudaFree(nets[i].ln1_b);
        if (nets[i].fc2_w) cudaFree(nets[i].fc2_w);
        if (nets[i].fc2_b) cudaFree(nets[i].fc2_b);
        if (nets[i].ln2_w) cudaFree(nets[i].ln2_w);
        if (nets[i].ln2_b) cudaFree(nets[i].ln2_b);
        for (int b = 0; b < 12; ++b) {
            if (nets[i].blocks[b].fc1_w) cudaFree(nets[i].blocks[b].fc1_w);
            if (nets[i].blocks[b].fc1_b) cudaFree(nets[i].blocks[b].fc1_b);
            if (nets[i].blocks[b].ln1_w) cudaFree(nets[i].blocks[b].ln1_w);
            if (nets[i].blocks[b].ln1_b) cudaFree(nets[i].blocks[b].ln1_b);
            if (nets[i].blocks[b].fc2_w) cudaFree(nets[i].blocks[b].fc2_w);
            if (nets[i].blocks[b].fc2_b) cudaFree(nets[i].blocks[b].fc2_b);
            if (nets[i].blocks[b].ln2_w) cudaFree(nets[i].blocks[b].ln2_w);
            if (nets[i].blocks[b].ln2_b) cudaFree(nets[i].blocks[b].ln2_b);
        }
        if (nets[i].fc_out_w) cudaFree(nets[i].fc_out_w);
        if (nets[i].fc_out_b) cudaFree(nets[i].fc_out_b);
    }

    // Zero device array
    MLPHead zero{};
    for (int i = 0; i < 32; ++i)
        cudaMemcpyToSymbol(d_networks, &zero, sizeof(MLPHead), i * sizeof(MLPHead), cudaMemcpyHostToDevice);
}

} // namespace NNUE_EVALUATE
