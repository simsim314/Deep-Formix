#ifndef MOVE_GENERATOR_BITBOARD_H
#define MOVE_GENERATOR_BITBOARD_H

#include "chess.h"
#include "cuda_adapters.cuh"          // <‑‑ gives CUDA_CALLABLE_MEMBER

class MoveGeneratorBitboard
{
public:
    // life‑cycle (host‑only is fine)
    static void init();
    static void destroy();

    // ── generateBoards ───────────────────────────────────────────────
#if USE_TEMPLATE_CHANCE_OPT == 1
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER                                   // ← add
    static uint32 generateBoards(HexaBitBoardPosition *pos,
                                 HexaBitBoardPosition *newPositions);
#else
    CUDA_CALLABLE_MEMBER
    static uint32 generateBoards(HexaBitBoardPosition *pos,
                                 HexaBitBoardPosition *newPositions,
                                 uint8 chance);
#endif

    // ── generateMoves ────────────────────────────────────────────────
#if USE_TEMPLATE_CHANCE_OPT == 1
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER
    static uint32 generateMoves(HexaBitBoardPosition *pos, CMove *genMoves);
#else
    CUDA_CALLABLE_MEMBER
    static uint32 generateMoves(HexaBitBoardPosition *pos, CMove *genMoves,
                                uint8 chance);
#endif

    // ── countMoves ───────────────────────────────────────────────────
#if USE_TEMPLATE_CHANCE_OPT == 1
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER
    static uint32 countMoves(HexaBitBoardPosition *pos);
#else
    CUDA_CALLABLE_MEMBER
    static uint32 countMoves(HexaBitBoardPosition *pos, uint8 chance);
#endif

    // ── makeMove ─────────────────────────────────────────────────────
#if USE_TEMPLATE_CHANCE_OPT == 1
    template <uint8 chance, bool updateHash>
    CUDA_CALLABLE_MEMBER
    static void makeMove(HexaBitBoardPosition *pos, uint64 &hash, CMove move);
#else
    CUDA_CALLABLE_MEMBER
    static void makeMove(HexaBitBoardPosition *pos, uint64 &hash, CMove move,
                         uint8 chance, bool updateHash);
#endif
};
#endif // MOVE_GENERATOR_BITBOARD_H
