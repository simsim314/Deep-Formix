#ifndef MOVE_ENCODER_H_
#define MOVE_ENCODER_H_

#include "chess.h"

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addCompactMove(uint32 *nMoves, CMove **genMoves, uint8 from, uint8 to, uint8 flags)
    {
        CMove move(from, to, flags);
        **genMoves = move;
        (*genMoves)++;
        (*nMoves)++;
    }
    // adds promotions if at promotion square
    // or normal pawn moves if not promotion
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addCompactPawnMoves(uint32 *nMoves, CMove **genMoves, uint8 from, uint64 dst, uint8 flags)
    {
        uint8 to = bitScan(dst);
        // promotion
        if (dst & (RANK1 | RANK8))
        {
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_KNIGHT_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_BISHOP_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_QUEEN_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_ROOK_PROMOTION);
        }
        else
        {
            addCompactMove(nMoves, genMoves, from, to, flags);
        }
    }

#endif // MOVE_ENCODER_H_
