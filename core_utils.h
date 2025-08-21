#ifndef CORE_UTILS_H_
#define CORE_UTILS_H_

#include "chess.h"
#include "bitboard_constants.h"

// This file contains the lowest-level utility functions that have no dependencies
// on higher-level logic like move generation or making.

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE void updateCastleFlag_internal(HexaBitBoardPosition *pos, uint64 dst, uint8 chance)
{
#if USE_BITWISE_MAGIC_FOR_CASTLE_FLAG_UPDATION == 1
    if (chance == WHITE) {
        pos->blackCastle &= ~( ((dst & BLACK_KING_SIDE_ROOK ) >> H8) | ((dst & BLACK_QUEEN_SIDE_ROOK) >> (A8-1))) ;
    } else {
        pos->whiteCastle &= ~( ((dst & WHITE_KING_SIDE_ROOK ) >> H1) | ((dst & WHITE_QUEEN_SIDE_ROOK) << 1)) ;
    }
#else
    if (chance == WHITE) {
        if (dst & BLACK_KING_SIDE_ROOK) pos->blackCastle &= ~CASTLE_FLAG_KING_SIDE;
        else if (dst & BLACK_QUEEN_SIDE_ROOK) pos->blackCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
    } else {
        if (dst & WHITE_KING_SIDE_ROOK) pos->whiteCastle &= ~CASTLE_FLAG_KING_SIDE;
        else if (dst & WHITE_QUEEN_SIDE_ROOK) pos->whiteCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
    }
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE void addMove_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *newBoard)
{
    **newPos = *newBoard;
    (*newPos)++;
    (*nMoves)++;
}

#endif // CORE_UTILS_H_
