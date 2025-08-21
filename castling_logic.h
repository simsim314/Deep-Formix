#ifndef CASTLING_LOGIC_H_
#define CASTLING_LOGIC_H_

#include "core_utils.h"

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE void addCastleMove_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos, uint64 kingFrom, uint64 kingTo, uint64 rookFrom, uint64 rookTo, uint8 chance)
{
    HexaBitBoardPosition newBoard;
    newBoard.bishopQueens = pos->bishopQueens;
    newBoard.pawns = pos->pawns;
    newBoard.knights = pos->knights;
    newBoard.kings = (pos->kings ^ kingFrom) | kingTo;
    newBoard.rookQueens = (pos->rookQueens ^ rookFrom) | rookTo;
    newBoard.chance = !chance;
    newBoard.enPassent = 0;
    newBoard.halfMoveCounter = 0;
    if (chance == WHITE) {
        newBoard.whitePieces = (pos->whitePieces ^ (kingFrom | rookFrom)) | (kingTo | rookTo);
        newBoard.whiteCastle = 0;
    } else {
        newBoard.blackCastle = 0;
        newBoard.whitePieces = pos->whitePieces;
    }
    addMove_internal(nMoves, newPos, &newBoard);
}

#endif // CASTLING_LOGIC_H_
