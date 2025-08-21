#ifndef EP_LOGIC_H_
#define EP_LOGIC_H_

#include "core_utils.h"

CUDA_CALLABLE_MEMBER static void addEnPassentMove_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos, uint64 src, uint64 dst, uint8 chance)
{
    HexaBitBoardPosition newBoard;
    uint64 capturedPiece = (chance == WHITE) ? southOne(dst) : northOne(dst);
    newBoard.bishopQueens = pos->bishopQueens;
    newBoard.rookQueens   = pos->rookQueens;
    newBoard.knights      = pos->knights;
    newBoard.kings        = pos->kings;
    newBoard.pawns = (pos->pawns ^ (capturedPiece | src)) | dst;
    if (chance == WHITE) newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
    else newBoard.whitePieces  = pos->whitePieces  ^ capturedPiece;
    newBoard.chance = !chance;
    newBoard.halfMoveCounter = 0;
    newBoard.enPassent = 0;
    addMove_internal(nMoves, newPos, &newBoard);
}

#endif // EP_LOGIC_H_
