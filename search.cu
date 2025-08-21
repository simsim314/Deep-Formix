#include <algorithm>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "thrust_wrappers.h"
#include "search.cuh"
#include "evaluate.cuh"
#include "MoveGeneratorBitboard.h"
#include "utils.h"
#include "embedding_nnue.cuh"
#include "evaluate_nnue.cuh"

namespace SEARCH {

Searcher g_searcher;

// ─── Kernels ─────────────────────────────────────────────────────────────────
__global__ void generateMoveCountsKernel(
    HexaBitBoardPosition* parent_boards,
    unsigned int* move_counts,
    int num_boards)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boards) return;

    HexaBitBoardPosition* pos = &parent_boards[idx];
    if (pos->kings == 0) {
        move_counts[idx] = 0; // Invalid/padded boards have 0 children
        return;
    }

    uint32 num_moves;
    if (pos->chance == WHITE)
        num_moves = MoveGeneratorBitboard::countMoves<WHITE>(pos);
    else
        num_moves = MoveGeneratorBitboard::countMoves<BLACK>(pos);

    // If a position has no legal moves (checkmate/stalemate),
    // we will generate one "null" child. So, we report its move count as 1.
    if (num_moves == 0) {
        move_counts[idx] = 1;
    } else {
        move_counts[idx] = num_moves;
    }
}

__global__ void generateBoardsAndMovesCombinedKernel(
    HexaBitBoardPosition* parent_boards,
    HexaBitBoardPosition* child_boards,
    CMove*                child_moves,
    const unsigned int*   move_offsets,
    int                   num_parent_boards)
{
    int parent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (parent_idx >= num_parent_boards) return;

    HexaBitBoardPosition* pos = &parent_boards[parent_idx];
    if (pos->kings == 0) return;

    HexaBitBoardPosition* children_boards_out = &child_boards[move_offsets[parent_idx]];
    CMove*                children_moves_out  = &child_moves[move_offsets[parent_idx]];

    CMove moves[MAX_MOVES];
    uint32 num_moves;
    if (pos->chance == WHITE) {
        num_moves = MoveGeneratorBitboard::generateMoves<WHITE>(pos, moves);
    } else {
        num_moves = MoveGeneratorBitboard::generateMoves<BLACK>(pos, moves);
    }

    if (num_moves == 0) {
        // This is a terminal node. Generate a single null move and a copy of the board.
        children_moves_out[0] = CMove(0, 0, 0);
        children_boards_out[0] = *pos;
    } else {
        for (uint32 i = 0; i < num_moves; ++i) {
            children_moves_out[i] = moves[i];
            children_boards_out[i] = *pos; // copy parent
            uint64 dummy_hash = 0;
            if (pos->chance == WHITE) {
                MoveGeneratorBitboard::makeMove<WHITE, false>(&children_boards_out[i], dummy_hash, moves[i]);
            } else {
                MoveGeneratorBitboard::makeMove<BLACK, false>(&children_boards_out[i], dummy_hash, moves[i]);
            }
        }
    }
}

__global__ void evaluateLeafNodesKernel(
        const HexaBitBoardPosition* boards,
        int                         num_boards,
        int*                        scores)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boards) return;

    const HexaBitBoardPosition* board = &boards[idx];

    if (board->kings == 0) {
        scores[idx] = 0;
    } else {
        scores[idx] = EVALUATION::evaluatePosition_nnue(board);
    }
}

__global__ void minimaxKernel(
        int*                parent_scores,
        const int*          child_scores,
        const unsigned int* move_offsets,
        int                 num_parents,
        int                 num_children,
        bool                maximizing_player,
        int*                best_child_indices)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_parents) return;

    unsigned int start = move_offsets[pid];
    unsigned int end   = (pid == num_parents - 1) ? num_children
                                                  : move_offsets[pid + 1];

    if (start == end) {
        parent_scores[pid] = maximizing_player ? -INF : INF;
        if (best_child_indices) best_child_indices[pid] = 0;
        return;
    }

    int best_score      = maximizing_player ? -INF : INF;
    int best_child_idx  = 0;

    for (unsigned int i = start; i < end; ++i) {
        int s = child_scores[i];
        if (maximizing_player) {
            if (s > best_score) { best_score = s; best_child_idx = i - start; }
        } else {
            if (s < best_score) { best_score = s; best_child_idx = i - start; }
        }
    }
    parent_scores[pid] = best_score;
    if (best_child_indices) best_child_indices[pid] = best_child_idx;
}

// ─── Searcher Implementation ─────────────────────────────────────────────────
Searcher::Searcher()
: d_boards_A(nullptr), d_boards_B(nullptr), d_move_counts(nullptr),
  d_scores_A(nullptr), d_scores_B(nullptr),
  m_max_boards(0), m_is_initialized(false) {}

void Searcher::init(size_t requested_max_boards) {
    if (m_is_initialized) return;

    cudaSetDevice(0);
    EVALUATION::init_nnue();

    const size_t bytes_per_board =
          2 * sizeof(HexaBitBoardPosition)
        + 1 * sizeof(unsigned int)
        + 2 * sizeof(int);

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    const double headroom = 0.60;
    const size_t usable = static_cast<size_t>(free_bytes * headroom);

    const size_t cap_by_mem = (bytes_per_board > 0) ? (usable / bytes_per_board) : 0;
    m_max_boards = (requested_max_boards == 0)
                 ? cap_by_mem
                 : std::min(requested_max_boards, cap_by_mem);

    if (m_max_boards == 0) {
        std::fprintf(stderr, "Not enough free GPU memory. Need ~%zu bytes/board; usable=%zu bytes.\n",
                     bytes_per_board, usable);
        std::exit(EXIT_FAILURE);
    }

    printf("Searcher: free %.2f GB, total %.2f GB, usable %.2f GB. theo bytes/board=%zu, capacity=%zu boards.\n",
           free_bytes / (1024.0*1024.0*1024.0),
           total_bytes / (1024.0*1024.0*1024.0),
           usable / (1024.0*1024.0*1024.0),
           bytes_per_board, m_max_boards);

    size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);

    CHECK_ALLOC(cudaMalloc(&d_boards_A,    sizeof(HexaBitBoardPosition) * m_max_boards));
    CHECK_ALLOC(cudaMalloc(&d_boards_B,    sizeof(HexaBitBoardPosition) * m_max_boards));
    CHECK_ALLOC(cudaMalloc(&d_move_counts, sizeof(unsigned int)         * m_max_boards));
    CHECK_ALLOC(cudaMalloc(&d_scores_A,    sizeof(int)                  * m_max_boards));
    CHECK_ALLOC(cudaMalloc(&d_scores_B,    sizeof(int)                  * m_max_boards));

    size_t free_after = 0, total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);

    const size_t empirical_delta = (free_before > free_after) ? (free_before - free_after) : 0;
    const double empirical_bpb = (m_max_boards > 0) ? (double)empirical_delta / (double)m_max_boards : 0.0;

    printf("Searcher initialized with capacity for %zu boards.\n", m_max_boards);
    printf("Empirical bytes/board ~ %.0f (measured), theoretical bytes/board = %zu\n",
           empirical_bpb, bytes_per_board);

    m_is_initialized = true;
}

void Searcher::cleanup() {
    if (!m_is_initialized) return;

    cudaFree(d_boards_A);        d_boards_A = nullptr;
    cudaFree(d_boards_B);        d_boards_B = nullptr;
    cudaFree(d_move_counts);     d_move_counts = nullptr;
    cudaFree(d_scores_A);        d_scores_A = nullptr;
    cudaFree(d_scores_B);        d_scores_B = nullptr;

    EVALUATION::cleanup_nnue();
    m_is_initialized = false;
    printf("Searcher cleaned up.\n");
}

SearchResult Searcher::findBestMove(const HexaBitBoardPosition& position, int depth)
{
    if (!m_is_initialized) {
        std::fprintf(stderr, "Searcher::findBestMove called before init().\n");
        std::exit(EXIT_FAILURE);
    }

    SearchResult result;
    std::vector<int>            level_num_boards(depth + 1, 0);
    std::vector<unsigned int*>  d_level_move_offsets(depth, nullptr);
    std::vector<int*>           d_level_best_child_indices(depth, nullptr);
    std::vector<CMove*>         d_level_child_moves(depth, nullptr);

    CHECK_ALLOC(cudaMemcpy(d_boards_A, &position, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice));

    HexaBitBoardPosition* parent_boards = d_boards_A;
    HexaBitBoardPosition* child_boards  = d_boards_B;
    level_num_boards[0] = 1;

    unsigned long long total_nodes_generated = 0;

    int* d_root_child_scores = nullptr; // <-- NEW: device buffer to hold root-child scores

    // FORWARD PASS: Generate the tree
    for (int d = 0; d < depth; ++d) {
        int num_parents = level_num_boards[d];
        if (num_parents == 0) break;

        dim3 threads(MAX_THREADS);
        dim3 blocks((num_parents + MAX_THREADS - 1) / MAX_THREADS);

        generateMoveCountsKernel<<<blocks, threads>>>(parent_boards, d_move_counts, num_parents);
        cudaDeviceSynchronize();

        CHECK_ALLOC(cudaMalloc(&d_level_move_offsets[d], sizeof(unsigned int) * num_parents));
        run_exclusive_scan(d_move_counts, d_level_move_offsets[d], num_parents);

        unsigned int total_moves = run_reduce(d_move_counts, num_parents);

        if (total_moves >= m_max_boards) {
             printf("Warning: Search capacity exceeded at depth %d. Terminating early.\n", d);
             total_moves = 0;
        }

        level_num_boards[d + 1] = total_moves;
        total_nodes_generated += total_moves;

        if (total_moves == 0) {
            for (int i = d + 1; i <= depth; ++i) level_num_boards[i] = 0;
            break;
        }

        CHECK_ALLOC(cudaMalloc(&d_level_child_moves[d], sizeof(CMove) * total_moves));
        generateBoardsAndMovesCombinedKernel<<<blocks, threads>>>(
            parent_boards, child_boards, d_level_child_moves[d], d_level_move_offsets[d], num_parents);
        cudaDeviceSynchronize();

        std::swap(parent_boards, child_boards);
    }

    // BACKWARD PASS: Evaluate leaves and propagate scores
    const int num_leaf_nodes = level_num_boards[depth];
    if (num_leaf_nodes > 0) {
        if ((size_t)num_leaf_nodes > m_max_boards) {
            fprintf(stderr, "Logic error: leaf nodes exceed persistent capacity\n");
            std::exit(EXIT_FAILURE);
        }

        dim3 threads(MAX_THREADS);
        dim3 leaf_blocks((num_leaf_nodes + MAX_THREADS - 1) / MAX_THREADS);
        evaluateLeafNodesKernel<<<leaf_blocks, threads>>>(
            parent_boards, num_leaf_nodes, d_scores_A);
        cudaDeviceSynchronize();

        int* parent_scores = d_scores_B;
        int* child_scores  = d_scores_A;

        for (int d = depth - 1; d >= 0; --d) {
            const int num_parents = level_num_boards[d];
            const int num_children = level_num_boards[d + 1];
            const bool is_max = ((position.chance == WHITE && d % 2 == 0)
                              || (position.chance == BLACK && d % 2 != 0));
            dim3 blocks((num_parents + MAX_THREADS - 1) / MAX_THREADS);

            // BEFORE we call minimax for d==0, child_scores currently points to the
            // device array holding scores for the root-children (level 1). Copy them.
            if (d == 0 && num_children > 0) {
                // allocate buffer to hold root-child scores
                CHECK_ALLOC(cudaMalloc(&d_root_child_scores, sizeof(int) * num_children));
                // copy device->device (child_scores -> d_root_child_scores)
                CHECK_CUDA(cudaMemcpy(d_root_child_scores, child_scores, sizeof(int) * num_children, cudaMemcpyDeviceToDevice));
            }

            CHECK_ALLOC(cudaMalloc(&d_level_best_child_indices[d], sizeof(int) * num_parents));

            minimaxKernel<<<blocks, threads>>>(
                parent_scores, child_scores, d_level_move_offsets[d],
                num_parents, num_children, is_max, d_level_best_child_indices[d]);
            cudaDeviceSynchronize();

            std::swap(parent_scores, child_scores);
        }
        // After the loop, 'child_scores' points to the final score buffer for the root node.
        CHECK_ALLOC(cudaMemcpy(&result.score, child_scores, sizeof(int), cudaMemcpyDeviceToHost));
    }

    result.nodes_searched = total_nodes_generated;

    // Reconstruct PV (host-side) and also collect root-level moves (host copies)
    if (level_num_boards[1] > 0) {
        std::vector<std::vector<unsigned int>> h_level_move_offsets(depth);
        std::vector<std::vector<int>>          h_level_best_child_indices(depth);
        std::vector<std::vector<CMove>>        h_level_child_moves(depth);

        for (int d = 0; d < depth; ++d) {
            if (level_num_boards[d+1] > 0) {
                h_level_move_offsets[d].resize(level_num_boards[d]);
                h_level_best_child_indices[d].resize(level_num_boards[d]);
                h_level_child_moves[d].resize(level_num_boards[d+1]);
                cudaMemcpy(h_level_move_offsets[d].data(), d_level_move_offsets[d], sizeof(unsigned int) * level_num_boards[d],   cudaMemcpyDeviceToHost);
                cudaMemcpy(h_level_best_child_indices[d].data(), d_level_best_child_indices[d], sizeof(int) * level_num_boards[d], cudaMemcpyDeviceToHost);
                cudaMemcpy(h_level_child_moves[d].data(), d_level_child_moves[d], sizeof(CMove) * level_num_boards[d+1],           cudaMemcpyDeviceToHost);
            }
        }

        int parent_node_idx = 0;
        for (int d = 0; d < depth; ++d) {
            if (level_num_boards[d+1] == 0 || parent_node_idx >= (int)h_level_best_child_indices[d].size()) break;

            int best_move_local_idx = h_level_best_child_indices[d][parent_node_idx];
            int absolute_move_idx = h_level_move_offsets[d][parent_node_idx] + best_move_local_idx;
            if (absolute_move_idx >= (int)h_level_child_moves[d].size()) break;

            CMove pv_move = h_level_child_moves[d][absolute_move_idx];
            
            // Do not add the null move to the principal variation.
            if (pv_move.getFrom() == 0 && pv_move.getTo() == 0) break;

            result.principal_variation.push_back(pv_move);
            if (d == 0) result.best_move = pv_move;

            parent_node_idx = absolute_move_idx;
        }

        // --- NEW: gather root-level move list + scores ---
        // Copy the device root child scores (if allocated) to host and pair with the root moves.
        if (d_root_child_scores != nullptr) {
            int root_children = level_num_boards[1];
            std::vector<int> h_root_scores(root_children);
            CHECK_CUDA(cudaMemcpy(h_root_scores.data(), d_root_child_scores, sizeof(int) * root_children, cudaMemcpyDeviceToHost));

            // Build vector of (CMove, score) of length root_children
            std::vector<std::pair<CMove,int>> root_list;
            root_list.reserve(root_children);

            // root moves are stored in h_level_child_moves[0] at indices [0 .. root_children-1]
            for (int i = 0; i < root_children; ++i) {
                CMove mv = h_level_child_moves[0][i];
                int mvscore = h_root_scores[i];
                // skip null move if present
                if (mv.getFrom() == 0 && mv.getTo() == 0) continue;
                root_list.emplace_back(mv, mvscore);
            }

            // sort descending by score (best first)
            std::sort(root_list.begin(), root_list.end(), [](const std::pair<CMove,int>& a, const std::pair<CMove,int>& b){
                return a.second > b.second;
            });

            result.root_move_scores = std::move(root_list);
        }
    }

    // Free per-depth temporaries
    for (auto ptr : d_level_move_offsets)        if (ptr) cudaFree(ptr);
    for (auto ptr : d_level_best_child_indices)  if (ptr) cudaFree(ptr);
    for (auto ptr : d_level_child_moves)         if (ptr) cudaFree(ptr);

    // free root-child scores buffer
    if (d_root_child_scores) cudaFree(d_root_child_scores);

    return result;
}


} // namespace SEARCH
