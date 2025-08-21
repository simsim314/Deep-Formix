#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <vector>
#include <cstdint>
#include <cinttypes>

#include "macros.cuh"
#include "chess.h"

namespace SEARCH {

// Holds the complete result of a search operation.
struct SearchResult {
    CMove best_move;
    int score;
    std::vector<CMove> principal_variation;
    uint64_t nodes_searched;
    std::vector<std::pair<CMove,int>> root_move_scores;
    
    SearchResult() : score(0), nodes_searched(0) {}
};

class Searcher {
public:
    Searcher();

    // Allocates persistent GPU memory once.
    // If max_boards == 0, autosize from usable VRAM. Otherwise, clamp to that cap.
    void init(size_t max_boards = 25000000);

    void cleanup();

    SearchResult findBestMove(const HexaBitBoardPosition& position, int depth);

private:
    // Persistent buffers for the entire run
    HexaBitBoardPosition *d_boards_A, *d_boards_B;
    unsigned int         *d_move_counts;
    int                  *d_scores_A,  *d_scores_B;

    // NNUE workspace is no longer needed here.

    size_t               m_max_boards;
    bool                 m_is_initialized;
};

// Global instance of the searcher, accessible throughout the engine.
extern Searcher g_searcher;

// Optional convenience wrappers
inline void init(size_t max_boards = 0) { g_searcher.init(max_boards); }
inline void terminate() { g_searcher.cleanup(); }

// --- Helpers (unchanged) ---
inline void moveToString(const CMove& move, char* buffer) {
    int from = move.getFrom();
    int to = move.getTo();
    uint8 flags = move.getFlags();

    buffer[0] = (from % 8) + 'a';
    buffer[1] = (from / 8) + '1';
    buffer[2] = (to % 8) + 'a';
    buffer[3] = (to / 8) + '1';

    int next_char_idx = 4;
    if (flags >= CM_FLAG_PROMOTION) {
        if (flags == CM_FLAG_QUEEN_PROMOTION || flags == CM_FLAG_QUEEN_PROMO_CAP) buffer[next_char_idx++] = 'q';
        else if (flags == CM_FLAG_ROOK_PROMOTION || flags == CM_FLAG_ROOK_PROMO_CAP) buffer[next_char_idx++] = 'r';
        else if (flags == CM_FLAG_BISHOP_PROMOTION || flags == CM_FLAG_BISHOP_PROMO_CAP) buffer[next_char_idx++] = 'b';
        else if (flags == CM_FLAG_KNIGHT_PROMOTION || flags == CM_FLAG_KNIGHT_PROMO_CAP) buffer[next_char_idx++] = 'n';
    }

    buffer[next_char_idx] = '\0';
}

inline void printSearchResult(const SearchResult& result) {
    if (result.principal_variation.empty()) {
        printf("info depth 0 score cp %.2f nodes %" PRIu64 " string no legal moves found\n",
               result.score / 100.0, result.nodes_searched);
        return;
    }

    printf("info depth %zu score cp %.2f nodes %" PRIu64 " pv",
           result.principal_variation.size(),
           result.score / 100.0,
           result.nodes_searched);

    char move_buffer[6];
    for (const auto& move : result.principal_variation) {
        moveToString(move, move_buffer);
        printf(" %s", move_buffer);
    }
    printf("\n");
}

}  // namespace SEARCH

#endif  // SEARCH_H_INCLUDED
