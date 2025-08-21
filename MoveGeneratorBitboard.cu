// ──────────────────────────────────────────────────────────────────────────────
//  MoveGeneratorBitboard.cu   – host‑/device wrappers for core move logic
// ──────────────────────────────────────────────────────────────────────────────
#include "MoveGeneratorBitboard.h"
#include "move_maker.h"
#include "move_generator.h"
#include "magic_tables.h"
#include "movegen_device.cuh"

#include <ctime>
#include <cassert>

// ============================================================================
//  Public lifecycle helpers
// ============================================================================
void MoveGeneratorBitboard::init()
{
    initialize_move_generator_tables();
}

void MoveGeneratorBitboard::destroy()
{
    /* nothing to clean up right now */
}

// ============================================================================
//  generateBoards
// ============================================================================
#if USE_TEMPLATE_CHANCE_OPT == 1
template <uint8 chance>
CUDA_CALLABLE_MEMBER
uint32 MoveGeneratorBitboard::generateBoards(HexaBitBoardPosition *pos,
                                             HexaBitBoardPosition *newPos)
{
    return generateBoards_internal<chance>(pos, newPos);
}
// explicit instantiations for host+device
template CUDA_CALLABLE_MEMBER uint32
MoveGeneratorBitboard::generateBoards<WHITE>(HexaBitBoardPosition*,
                                             HexaBitBoardPosition*);
template CUDA_CALLABLE_MEMBER uint32
MoveGeneratorBitboard::generateBoards<BLACK>(HexaBitBoardPosition*,
                                             HexaBitBoardPosition*);
#else
CUDA_CALLABLE_MEMBER
uint32 MoveGeneratorBitboard::generateBoards(HexaBitBoardPosition *pos,
                                             HexaBitBoardPosition *newPos,
                                             uint8 chance)
{
    return (chance == WHITE)
           ? generateBoards_internal<WHITE>(pos, newPos)
           : generateBoards_internal<BLACK>(pos, newPos);
}
#endif

// ============================================================================
//  generateMoves
// ============================================================================
#if USE_TEMPLATE_CHANCE_OPT == 1
template <uint8 chance>
CUDA_CALLABLE_MEMBER
uint32 MoveGeneratorBitboard::generateMoves(HexaBitBoardPosition *pos,
                                            CMove *genMoves)
{
    return generateMoves_internal<chance>(pos, genMoves);
}
template CUDA_CALLABLE_MEMBER uint32
MoveGeneratorBitboard::generateMoves<WHITE>(HexaBitBoardPosition*, CMove*);
template CUDA_CALLABLE_MEMBER uint32
MoveGeneratorBitboard::generateMoves<BLACK>(HexaBitBoardPosition*, CMove*);
#else
CUDA_CALLABLE_MEMBER
uint32 MoveGeneratorBitboard::generateMoves(HexaBitBoardPosition *pos,
                                            CMove *genMoves,
                                            uint8 chance)
{
    return (chance == WHITE)
           ? generateMoves_internal<WHITE>(pos, genMoves)
           : generateMoves_internal<BLACK>(pos, genMoves);
}
#endif

// ============================================================================
//  countMoves
// ============================================================================
#if USE_TEMPLATE_CHANCE_OPT == 1
template <uint8 chance>
CUDA_CALLABLE_MEMBER
uint32 MoveGeneratorBitboard::countMoves(HexaBitBoardPosition *pos)
{
    return countMoves_internal<chance>(pos);
}
template CUDA_CALLABLE_MEMBER uint32
MoveGeneratorBitboard::countMoves<WHITE>(HexaBitBoardPosition*);
template CUDA_CALLABLE_MEMBER uint32
MoveGeneratorBitboard::countMoves<BLACK>(HexaBitBoardPosition*);
#else
CUDA_CALLABLE_MEMBER
uint32 MoveGeneratorBitboard::countMoves(HexaBitBoardPosition *pos,
                                         uint8 chance)
{
    return (chance == WHITE)
           ? countMoves_internal<WHITE>(pos)
           : countMoves_internal<BLACK>(pos);
}
#endif

// ============================================================================
//  makeMove
// ============================================================================
#if USE_TEMPLATE_CHANCE_OPT == 1
template <uint8 chance, bool updateHash>
CUDA_CALLABLE_MEMBER
void MoveGeneratorBitboard::makeMove(HexaBitBoardPosition *pos,
                                     uint64 &hash,
                                     CMove move)
{
    makeMove_internal<chance, updateHash>(pos, hash, move);
}
template CUDA_CALLABLE_MEMBER void
MoveGeneratorBitboard::makeMove<WHITE, true >(HexaBitBoardPosition*, uint64&, CMove);
template CUDA_CALLABLE_MEMBER void
MoveGeneratorBitboard::makeMove<WHITE, false>(HexaBitBoardPosition*, uint64&, CMove);
template CUDA_CALLABLE_MEMBER void
MoveGeneratorBitboard::makeMove<BLACK, true >(HexaBitBoardPosition*, uint64&, CMove);
template CUDA_CALLABLE_MEMBER void
MoveGeneratorBitboard::makeMove<BLACK, false>(HexaBitBoardPosition*, uint64&, CMove);
#else
CUDA_CALLABLE_MEMBER
void MoveGeneratorBitboard::makeMove(HexaBitBoardPosition *pos,
                                     uint64 &hash,
                                     CMove   move,
                                     uint8   chance,
                                     bool    updateHash)
{
    if (chance == WHITE)
        updateHash ? makeMove_internal<WHITE, true >(pos, hash, move)
                   : makeMove_internal<WHITE, false>(pos, hash, move);
    else
        updateHash ? makeMove_internal<BLACK, true >(pos, hash, move)
                   : makeMove_internal<BLACK, false>(pos, hash, move);
}
#endif
