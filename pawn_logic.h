#ifndef PAWN_LOGIC_H_
#define PAWN_LOGIC_H_

#include "core_utils.h"

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE void addSinglePawnMove_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos, uint64 src, uint64 dst, uint8 chance, bool doublePush, uint8 pawnIndex)
{
    HexaBitBoardPosition newBoard;
    newBoard.bishopQueens = pos->bishopQueens & ~dst;
    newBoard.rookQueens   = pos->rookQueens   & ~dst;
    newBoard.knights      = pos->knights      & ~dst;
    newBoard.kings        = pos->kings;
    newBoard.pawns = (pos->pawns ^ src) | dst;
    if (chance == WHITE) newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
    else newBoard.whitePieces  = pos->whitePieces  & ~dst;
    newBoard.chance = !chance;
    if (doublePush) newBoard.enPassent = (pawnIndex & 7) + 1;
    else newBoard.enPassent = 0;
    newBoard.halfMoveCounter = 0;
    addMove_internal(nMoves, newPos, &newBoard);
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE void addPawnMoves_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos, uint64 src, uint64 dst, uint8 chance)
{
    if (dst & (RANK1 | RANK8)) {
        HexaBitBoardPosition newBoard;
        newBoard.kings = pos->kings;
        if (chance == WHITE) newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
        else newBoard.whitePieces  = pos->whitePieces  & ~dst;
        newBoard.pawns = (pos->pawns ^ src);
        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        newBoard.halfMoveCounter = 0;
        updateCastleFlag_internal(&newBoard, dst, chance);
        newBoard.knights      = pos->knights      | dst;
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        addMove_internal(nMoves, newPos, &newBoard);
        newBoard.knights      = pos->knights      & ~dst;
        newBoard.bishopQueens = pos->bishopQueens | dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        addMove_internal(nMoves, newPos, &newBoard);
        newBoard.rookQueens   = pos->rookQueens   | dst;
        addMove_internal(nMoves, newPos, &newBoard);
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        addMove_internal(nMoves, newPos, &newBoard);
    } else {
        addSinglePawnMove_internal(nMoves, newPos, pos, src, dst, chance, false, 0);
    }
}

#endif // PAWN_LOGIC_H_
