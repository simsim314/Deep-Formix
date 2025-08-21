// src/embedding_nnue.cu
#include "embedding_nnue.cuh"
#include "bitboard_constants.h"
#include "bitboard_utils.h"
#include "macros.cuh"
#include "switches.h" 
#include "uci.cuh" // For access to global network path

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace NNUE_EMBEDDING {

// ============================================================================
// DEVICE-SIDE GLOBAL VARIABLES
// ============================================================================
__device__ BucketEmbedding d_embeddings[32];

// Bucket mapping (device-side)
__device__ uint64_t* d_bucket_keys   = nullptr;
__device__ int32_t*  d_bucket_values = nullptr;
__device__ int       g_num_bucket_entries = 0;

// Host-side handles (to free during cleanup)
static uint64_t* h_d_bucket_keys   = nullptr;
static int32_t*  h_d_bucket_values = nullptr;

// ============================================================================
// DEVICE-SIDE IMPLEMENTATION
// ============================================================================
__device__ uint64_t get_map_key_device(const HexaBitBoardPosition* board) {
    uint64 queens  = board->bishopQueens & board->rookQueens;
    uint64 rooks   = board->rookQueens   & ~queens;
    uint64 bishops = board->bishopQueens & ~queens;
    uint64 pawns   = board->pawns & RANKS2TO7;

    uint64_t has_queen_flag = (queens != 0) ? 1ull : 0ull;
    uint64_t piece_count = (uint64_t)popCount(rooks)
                         + (uint64_t)popCount(board->knights)
                         + (uint64_t)popCount(bishops)
                         + (uint64_t)popCount(queens);
    uint64_t pawn_count = (uint64_t)popCount(pawns);

    return (has_queen_flag << 56) | (piece_count << 48) | (pawn_count << 40);
}

__device__ int get_bucket_index(const HexaBitBoardPosition* board) {
    uint64_t key = get_map_key_device(board);
    if (!d_bucket_keys || !d_bucket_values || g_num_bucket_entries <= 0) return 0;
    for (int i = 0; i < g_num_bucket_entries; ++i) {
        if (d_bucket_keys[i] == key) return d_bucket_values[i];
    }
    return 0;
}

__device__ void generate_embedding(float* x_emb, const HexaBitBoardPosition* board, int bucket_idx) {
    const bool is_white_turn = (board->chance == WHITE);
    const BucketEmbedding& emb = d_embeddings[bucket_idx];

    // Inverted selection to match the CPU estimator’s behavior
    const float* piece_table  = is_white_turn ? emb.W_black_piece  : emb.W_white_piece;
    const float* castle_table = is_white_turn ? emb.W_black_castle : emb.W_white_castle;
    const float* ep_table     = is_white_turn ? emb.W_black_ep     : emb.W_white_ep;
    const float* fifty_table  = is_white_turn ? emb.W_black_fifty  : emb.W_white_fifty;

    float* pieces_vec = &x_emb[0];
    for (int i = 0; i < 2048; ++i) pieces_vec[i] = 0.0f;

    uint64 all_pieces = (board->pawns & RANKS2TO7) | board->knights | board->bishopQueens | board->rookQueens | board->kings;
    uint64 queens = board->bishopQueens & board->rookQueens;

    uint64 tmp = all_pieces;
    while (tmp) {
        uint64 piece_bb = getOne(tmp);
        int sq_idx = bitScan(piece_bb);
        int piece_type_idx = -1;
        if      (piece_bb & (board->pawns & RANKS2TO7))             piece_type_idx = 0;
        else if (piece_bb & board->knights)                         piece_type_idx = 1;
        else if (piece_bb & (board->bishopQueens & ~queens))        piece_type_idx = 2;
        else if (piece_bb & (board->rookQueens   & ~queens))        piece_type_idx = 3;
        else if (piece_bb & queens)                                  piece_type_idx = 4;
        else if (piece_bb & board->kings)                            piece_type_idx = 5;
        int color_offset = (piece_bb & board->whitePieces) ? 0 : 6;
        int piece_idx = piece_type_idx + color_offset;
        int table_row = sq_idx * 12 + piece_idx;
        for (int k = 0; k < 32; ++k)
            pieces_vec[sq_idx * 32 + k] = piece_table[table_row * 32 + k];
        tmp ^= piece_bb;
    }

    float* castle_vec = &x_emb[2048];
    float castle_rights[4] = {
        (float)((board->whiteCastle & CASTLE_FLAG_KING_SIDE ) != 0),
        (float)((board->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) != 0),
        (float)((board->blackCastle & CASTLE_FLAG_KING_SIDE ) != 0),
        (float)((board->blackCastle & CASTLE_FLAG_QUEEN_SIDE) != 0)
    };
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 32; ++k)
            castle_vec[i * 32 + k] = castle_rights[i] * castle_table[i * 32 + k];

    float* ep_vec = &x_emb[2176];
    for (int k = 0; k < 32; ++k) ep_vec[k] = 0.0f;
    if (board->enPassent != 0) {
        int file_idx = board->enPassent - 1;
        for (int k = 0; k < 32; ++k)
            ep_vec[k] = ep_table[file_idx * 32 + k];
    }

    float* fifty_vec = &x_emb[2208];
    float fifty_a = fminf((float)board->halfMoveCounter / 100.0f, 1.0f);
    for (int k = 0; k < 32; ++k)
        fifty_vec[k] = (1.0f - fifty_a) * fifty_table[0 * 32 + k] + fifty_a * fifty_table[1 * 32 + k];
}

// ============================================================================
// HOST-SIDE INITIALIZATION
// ============================================================================
void init(const float* model_data) {
    const float* current_pos = model_data;

    // Per-bucket embedding sizes (floats)
    const size_t EMB_WHITE_PIECE  = 64 * 12 * 32;
    const size_t EMB_BLACK_PIECE  = 64 * 12 * 32;
    const size_t EMB_WHITE_CASTLE = 4 * 32;
    const size_t EMB_BLACK_CASTLE = 4 * 32;
    const size_t EMB_WHITE_EP     = 8 * 32;
    const size_t EMB_BLACK_EP     = 8 * 32;
    const size_t EMB_WHITE_FIFTY  = 2 * 32;
    const size_t EMB_BLACK_FIFTY  = 2 * 32;

    // Network size to skip (floats) — matches evaluate_nnue.cu loader
    const size_t NET_FC1_W = 256 * 2240;
    const size_t NET_FC1_B = 256;
    const size_t NET_LN1_W = 256;
    const size_t NET_LN1_B = 256;
    const size_t NET_FC2_W = 32 * 256;
    const size_t NET_FC2_B = 32;
    const size_t NET_LN2_W = 32;
    const size_t NET_LN2_B = 32;
    const size_t RBLOCK    = (32*32) + 32 + 32 + 32 + (32*32) + 32 + 32 + 32; // fc1_w, fc1_b, ln1_w, ln1_b, fc2_w, fc2_b, ln2_w, ln2_b
    const size_t NET_BLOCKS = 12 * RBLOCK;
    const size_t NET_FCOUT_W = 51 * 32;
    const size_t NET_FCOUT_B = 51;
    const size_t NET_SIZE = NET_FC1_W + NET_FC1_B + NET_LN1_W + NET_LN1_B
                          + NET_FC2_W + NET_FC2_B + NET_LN2_W + NET_LN2_B
                          + NET_BLOCKS + NET_FCOUT_W + NET_FCOUT_B;

    for (int i = 0; i < 32; ++i) {
        BucketEmbedding d_emb_staging;

        // Embedding tensors
        cudaMalloc(&d_emb_staging.W_white_piece,  sizeof(float) * EMB_WHITE_PIECE);
        cudaMemcpy(d_emb_staging.W_white_piece,   current_pos,   sizeof(float) * EMB_WHITE_PIECE,  cudaMemcpyHostToDevice);
        current_pos += EMB_WHITE_PIECE;

        cudaMalloc(&d_emb_staging.W_black_piece,  sizeof(float) * EMB_BLACK_PIECE);
        cudaMemcpy(d_emb_staging.W_black_piece,   current_pos,   sizeof(float) * EMB_BLACK_PIECE,  cudaMemcpyHostToDevice);
        current_pos += EMB_BLACK_PIECE;

        cudaMalloc(&d_emb_staging.W_white_castle, sizeof(float) * EMB_WHITE_CASTLE);
        cudaMemcpy(d_emb_staging.W_white_castle,  current_pos,   sizeof(float) * EMB_WHITE_CASTLE, cudaMemcpyHostToDevice);
        current_pos += EMB_WHITE_CASTLE;

        cudaMalloc(&d_emb_staging.W_black_castle, sizeof(float) * EMB_BLACK_CASTLE);
        cudaMemcpy(d_emb_staging.W_black_castle,  current_pos,   sizeof(float) * EMB_BLACK_CASTLE, cudaMemcpyHostToDevice);
        current_pos += EMB_BLACK_CASTLE;

        cudaMalloc(&d_emb_staging.W_white_ep,     sizeof(float) * EMB_WHITE_EP);
        cudaMemcpy(d_emb_staging.W_white_ep,      current_pos,   sizeof(float) * EMB_WHITE_EP,     cudaMemcpyHostToDevice);
        current_pos += EMB_WHITE_EP;

        cudaMalloc(&d_emb_staging.W_black_ep,     sizeof(float) * EMB_BLACK_EP);
        cudaMemcpy(d_emb_staging.W_black_ep,      current_pos,   sizeof(float) * EMB_BLACK_EP,     cudaMemcpyHostToDevice);
        current_pos += EMB_BLACK_EP;

        cudaMalloc(&d_emb_staging.W_white_fifty,  sizeof(float) * EMB_WHITE_FIFTY);
        cudaMemcpy(d_emb_staging.W_white_fifty,   current_pos,   sizeof(float) * EMB_WHITE_FIFTY,  cudaMemcpyHostToDevice);
        current_pos += EMB_WHITE_FIFTY;

        cudaMalloc(&d_emb_staging.W_black_fifty,  sizeof(float) * EMB_BLACK_FIFTY);
        cudaMemcpy(d_emb_staging.W_black_fifty,   current_pos,   sizeof(float) * EMB_BLACK_FIFTY,  cudaMemcpyHostToDevice);
        current_pos += EMB_BLACK_FIFTY;

        // Skip network weights for this bucket (they are loaded by NNUE_EVALUATE::init)
        current_pos += NET_SIZE;

        // Commit this bucket’s embedding pointers to device
        cudaMemcpyToSymbol(d_embeddings, &d_emb_staging, sizeof(BucketEmbedding),
                           i * sizeof(BucketEmbedding), cudaMemcpyHostToDevice);
    }

    // Load mapping using the global variable
    std::ifstream map_file(UCI::g_nnue_mapping_path, std::ios::binary | std::ios::ate);
    
    if (!map_file) {
        fprintf(stderr, "Could not open mapping file: %s\n", UCI::g_nnue_mapping_path.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::streamsize sz = map_file.tellg();
    map_file.seekg(0, std::ios::beg);
    int num_entries = static_cast<int>(sz / (sizeof(uint64_t) + sizeof(int32_t)));
    cudaMemcpyToSymbol(g_num_bucket_entries, &num_entries, sizeof(int));

    std::vector<uint64_t> h_keys(num_entries);
    std::vector<int32_t>  h_vals(num_entries);
    for (int i = 0; i < num_entries; ++i) {
        map_file.read(reinterpret_cast<char*>(&h_keys[i]), sizeof(uint64_t));
        map_file.read(reinterpret_cast<char*>(&h_vals[i]), sizeof(int32_t));
    }

    // Allocate device arrays and copy data
    cudaMalloc(&h_d_bucket_keys,   num_entries * sizeof(uint64_t));
    cudaMemcpy(h_d_bucket_keys, h_keys.data(), num_entries * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaMalloc(&h_d_bucket_values, num_entries * sizeof(int32_t));
    cudaMemcpy(h_d_bucket_values, h_vals.data(), num_entries * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Set device pointer symbols
    cudaMemcpyToSymbol(d_bucket_keys,   &h_d_bucket_keys,   sizeof(h_d_bucket_keys));
    cudaMemcpyToSymbol(d_bucket_values, &h_d_bucket_values, sizeof(h_d_bucket_values));

    std::cout << "NNUE embeddings and mapping loaded to GPU. (" << num_entries << " entries)\n";
}

void cleanup() {
    // Free mapping arrays
    uint64_t* d_keys = nullptr;
    int32_t*  d_vals = nullptr;
    cudaMemcpyFromSymbol(&d_keys, d_bucket_keys,   sizeof(d_keys));
    cudaMemcpyFromSymbol(&d_vals, d_bucket_values, sizeof(d_vals));
    if (d_keys) cudaFree(d_keys);
    if (d_vals) cudaFree(d_vals);
    d_keys = nullptr; d_vals = nullptr;
    int zero = 0;
    cudaMemcpyToSymbol(d_bucket_keys,   &d_keys, sizeof(d_keys));
    cudaMemcpyToSymbol(d_bucket_values, &d_vals, sizeof(d_vals));
    cudaMemcpyToSymbol(g_num_bucket_entries, &zero, sizeof(int));
    h_d_bucket_keys = nullptr;
    h_d_bucket_values = nullptr;

    // Free embedding tensors
    BucketEmbedding buckets[32];
    cudaMemcpyFromSymbol(buckets, d_embeddings, sizeof(buckets));
    for (int i = 0; i < 32; ++i) {
        if (buckets[i].W_white_piece)   cudaFree(buckets[i].W_white_piece);
        if (buckets[i].W_black_piece)   cudaFree(buckets[i].W_black_piece);
        if (buckets[i].W_white_castle)  cudaFree(buckets[i].W_white_castle);
        if (buckets[i].W_black_castle)  cudaFree(buckets[i].W_black_castle);
        if (buckets[i].W_white_ep)      cudaFree(buckets[i].W_white_ep);
        if (buckets[i].W_black_ep)      cudaFree(buckets[i].W_black_ep);
        if (buckets[i].W_white_fifty)   cudaFree(buckets[i].W_white_fifty);
        if (buckets[i].W_black_fifty)   cudaFree(buckets[i].W_black_fifty);
    }
    BucketEmbedding zero_emb{};
    for (int i = 0; i < 32; ++i)
        cudaMemcpyToSymbol(d_embeddings, &zero_emb, sizeof(BucketEmbedding),
                           i * sizeof(BucketEmbedding), cudaMemcpyHostToDevice);
}

} // namespace NNUE_EMBEDDING
