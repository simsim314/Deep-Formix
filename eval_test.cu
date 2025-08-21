#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cinttypes>

#include "MoveGeneratorBitboard.h"
#include "utils.h"
#include "evaluate.cuh"
#include "macros.cuh"
#include "embedding_nnue.cuh"
#include "evaluate_nnue.cuh"
#include "search.cuh"
#include "bitboard_utils.h"

// Kernel to evaluate a batch of board positions using NNUE
__global__ void evaluate_positions_kernel(
    HexaBitBoardPosition* boards,
    int* scores,
    int num_boards)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boards) return;

    scores[idx] = EVALUATION::evaluatePosition_nnue(&boards[idx]);
}

// Struct to hold a move and its score for sorting
struct ScoredMove {
    CMove move;
    int score;

    bool operator<(const ScoredMove& other) const {
        return score > other.score; // Sort descending
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " \"<FEN string>\"" << std::endl;
        return 1;
    }

    std::cout << "Initializing engine components..." << std::endl;
    MoveGeneratorBitboard::init();
    EVALUATION::init_nnue();

    std::string fen = argv[1];
    std::cout << "Evaluating FEN: " << fen << std::endl;
    HexaBitBoardPosition h_initial_board;
    BoardPosition pos088;
    Utils::readFENString(const_cast<char*>(fen.c_str()), &pos088);
    Utils::board088ToHexBB(&h_initial_board, &pos088);
    Utils::displayBoard(&h_initial_board);
    
    // Generate legal moves from the initial position
    std::vector<CMove> h_moves(MAX_MOVES);
    uint32 num_moves;
    if (h_initial_board.chance == WHITE) {
        num_moves = MoveGeneratorBitboard::generateMoves<WHITE>(&h_initial_board, h_moves.data());
    } else {
        num_moves = MoveGeneratorBitboard::generateMoves<BLACK>(&h_initial_board, h_moves.data());
    }
    h_moves.resize(num_moves);

    // Create a vector to hold all board positions to be evaluated
    std::vector<HexaBitBoardPosition> h_boards_to_eval;
    h_boards_to_eval.reserve(num_moves + 1);

    // Add the initial board itself to the list as the first element
    h_boards_to_eval.push_back(h_initial_board);

    // Generate all child boards resulting from legal moves and add them
    for (uint32 i = 0; i < num_moves; ++i) {
        HexaBitBoardPosition child_board = h_initial_board;
        uint64 dummy_hash = 0;
        if (h_initial_board.chance == WHITE) {
            MoveGeneratorBitboard::makeMove<WHITE, false>(&child_board, dummy_hash, h_moves[i]);
        } else {
            MoveGeneratorBitboard::makeMove<BLACK, false>(&child_board, dummy_hash, h_moves[i]);
        }
        h_boards_to_eval.push_back(child_board);
    }
    
    const int total_boards_to_eval = h_boards_to_eval.size();

    if (total_boards_to_eval == 0) {
        std::cout << "No valid board to evaluate." << std::endl;
        EVALUATION::cleanup_nnue();
        return 0;
    }

    // Allocate GPU memory for all boards and their scores
    HexaBitBoardPosition* d_boards;
    int* d_scores;
    cudaMalloc(&d_boards, total_boards_to_eval * sizeof(HexaBitBoardPosition));
    cudaMalloc(&d_scores, total_boards_to_eval * sizeof(int));

    // Copy all boards (parent + children) to the GPU
    cudaMemcpy(d_boards, h_boards_to_eval.data(), total_boards_to_eval * sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);

    // Launch a single kernel to evaluate all boards in the batch
    dim3 threads(256);
    dim3 blocks((total_boards_to_eval + threads.x - 1) / threads.x);
    std::cout << "\nLaunching kernel to evaluate " << total_boards_to_eval << " positions (initial + children)..." << std::endl;
    evaluate_positions_kernel<<<blocks, threads>>>(d_boards, d_scores, total_boards_to_eval);
    cudaDeviceSynchronize();

    // Copy all scores back to the host
    std::vector<int> h_scores(total_boards_to_eval);
    cudaMemcpy(h_scores.data(), d_scores, total_boards_to_eval * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Print Results ---

    // The first score corresponds to the initial position
    int initial_score = h_scores[0];
    std::cout << "\n--- Evaluation of Initial Position ---" << std::endl;
    printf("Initial Position Score: %6d\n", initial_score);

    if (num_moves == 0) {
        std::cout << "\nNo legal moves found from this position." << std::endl;
    } else {
        // Populate the scored moves list (child scores start from index 1 of h_scores)
        std::vector<ScoredMove> scored_moves(num_moves);
        for (uint32 i = 0; i < num_moves; ++i) {
            scored_moves[i] = {h_moves[i], h_scores[i + 1]};
        }
        std::sort(scored_moves.begin(), scored_moves.end());

        std::cout << "\n--- Sorted Evaluations of Child Positions (from White's perspective) ---" << std::endl;
        char move_buffer[6];
        for (const auto& sm : scored_moves) {
            SEARCH::moveToString(sm.move, move_buffer);
            printf("%-8s | Score: %6d\n", move_buffer, sm.score);
        }
    }

    // --- Cleanup ---
    cudaFree(d_boards);
    cudaFree(d_scores);
    EVALUATION::cleanup_nnue();

    return 0;
}
