#ifndef MOVE_MAKER_H_
#define MOVE_MAKER_H_

#include "bitboard_utils.h"
#include "chess.h"
#include "cuda_adapters.cuh"  // ADDED for ZOB_KEY macros and other adapters
#include "magic_tables.h"     // ADDED for zob, zob2 host variables

// MOVED FROM castling_logic.h to break include cycle
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void updateCastleFlag(HexaBitBoardPosition *pos, uint64 dst, uint8 chance)
{
#if USE_BITWISE_MAGIC_FOR_CASTLE_FLAG_UPDATION == 1
    if (chance == WHITE)
    {
        pos->blackCastle &= ~( ((dst & BLACK_KING_SIDE_ROOK ) >> H8)      |
                               ((dst & BLACK_QUEEN_SIDE_ROOK) >> (A8-1))) ;
    }
    else
    {
        pos->whiteCastle &= ~( ((dst & WHITE_KING_SIDE_ROOK ) >> H1) |
                               ((dst & WHITE_QUEEN_SIDE_ROOK) << 1)) ;
    }
#else
    if (chance == WHITE)
    {
        if (dst & BLACK_KING_SIDE_ROOK)
            pos->blackCastle &= ~CASTLE_FLAG_KING_SIDE;
        else if (dst & BLACK_QUEEN_SIDE_ROOK)
            pos->blackCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
    }
    else
    {
        if (dst & WHITE_KING_SIDE_ROOK)
            pos->whiteCastle &= ~CASTLE_FLAG_KING_SIDE;
        else if (dst & WHITE_QUEEN_SIDE_ROOK)
            pos->whiteCastle &= ~CASTLE_FLAG_QUEEN_SIDE;
    }
#endif
}


    // adds the given board to list and increments the move counter
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addMove(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *newBoard)
    {
        **newPos = *newBoard;
        (*newPos)++;
        (*nMoves)++;
    }
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addSlidingMove_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                             uint64 src, uint64 dst, uint8 chance)
    {

#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = !!((pos->bishopQueens | pos->rookQueens | pos->knights | (pos->pawns & RANKS2TO7) | pos->knights) & dst);
            // Utils::displayMoveBB(move); // Utils not available here
        }
#endif

        HexaBitBoardPosition newBoard;

        // remove the dst from all bitboards
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        newBoard.kings        = pos->kings        & ~dst;
        newBoard.knights      = pos->knights      & ~dst;
        newBoard.pawns        = pos->pawns        & ~(dst & RANKS2TO7);

        // figure out if the piece was a bishop, rook, or a queen
        uint64 isBishop = newBoard.bishopQueens & src;
        uint64 isRook   = newBoard.rookQueens   & src;

        // remove src from the appropriate board / boards if queen
        newBoard.bishopQueens ^= isBishop;
        newBoard.rookQueens   ^= isRook;

        // add dst
        newBoard.bishopQueens |= isBishop ? dst : 0;
        newBoard.rookQueens   |= isRook   ? dst : 0;

        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  & ~dst;
        }

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        //newBoard.halfMoveCounter++;   // quiet move -> increment half move counter // TODO: correctly increment this based on if there was a capture

        // need to update castle flag for both sides (src moved in same side, and dst move on other side)
        updateCastleFlag(&newBoard, dst,  chance);
        updateCastleFlag(&newBoard, src, !chance);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addKnightMove_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                            uint64 src, uint64 dst, uint8 chance)
    {
#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = !!((pos->bishopQueens | pos->knights | (pos->pawns & RANKS2TO7) | pos->knights | pos->rookQueens) & dst);
            // Utils::displayMoveBB(move); // Utils not available here
        }
#endif
        HexaBitBoardPosition newBoard;

        // remove the dst from all bitboards
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        newBoard.kings        = pos->kings        & ~dst;
        newBoard.pawns        = pos->pawns        & ~(dst & RANKS2TO7);

        // remove src and add destination
        newBoard.knights      = (pos->knights ^ src) | dst;

        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  & ~dst;
        }

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        //newBoard.halfMoveCounter++;   // quiet move -> increment half move counter // TODO: correctly increment this based on if there was a capture
        updateCastleFlag(&newBoard, dst, chance);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addKingMove_internal(uint32 *nMoves, HexaBitBoardPosition **newPos, HexaBitBoardPosition *pos,
                                          uint64 src, uint64 dst, uint8 chance)
    {
#if DEBUG_PRINT_MOVES == 1
        if (printMoves)
        {
            Move move;
            move.src = bitScan(src);
            move.dst = bitScan(dst);
            move.flags = 0;
            move.capturedPiece = !!((pos->bishopQueens | pos->knights | (pos->pawns & RANKS2TO7) | pos->knights | pos->rookQueens) & dst);
            // Utils::displayMoveBB(move); // Utils not available here
        }
#endif
        HexaBitBoardPosition newBoard;

        // remove the dst from all bitboards
        newBoard.bishopQueens = pos->bishopQueens & ~dst;
        newBoard.rookQueens   = pos->rookQueens   & ~dst;
        newBoard.knights      = pos->knights      & ~dst;
        newBoard.pawns        = pos->pawns        & ~(dst & RANKS2TO7);

        // remove src and add destination
        newBoard.kings = (pos->kings ^ src) | dst;

        if (chance == WHITE)
        {
            newBoard.whitePieces = (pos->whitePieces ^ src) | dst;
            newBoard.whiteCastle = 0;
        }
        else
        {
            newBoard.whitePieces  = pos->whitePieces  & ~dst;
            newBoard.blackCastle = 0;
        }

        // update game state (the old game state already got copied over above when copying pawn bitboard)
        newBoard.chance = !chance;
        newBoard.enPassent = 0;
        // newBoard.halfMoveCounter++;   // quiet move -> increment half move counter (TODO: fix this for captures)
        updateCastleFlag(&newBoard, dst, chance);

        // add the move
        addMove(nMoves, newPos, &newBoard);
    }
#if USE_TEMPLATE_CHANCE_OPT == 1
    template<uint8 chance, bool updateHash>
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void makeMove_internal (HexaBitBoardPosition *pos, uint64 &hash, CMove move)
#else
	CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void makeMove_internal(HexaBitBoardPosition *pos, uint64 &hash, CMove move, uint8 chance, bool updateHash)
#endif
    {
        uint64 src = BIT(move.getFrom());
        uint64 dst = BIT(move.getTo());

        // figure out the source piece
        uint64 queens = pos->bishopQueens & pos->rookQueens;
        uint8 piece = 0;
        if (pos->kings & src)
            piece = KING;
        else if (pos->knights & src)
            piece = KNIGHT;
        else if ((pos->pawns & RANKS2TO7) & src)
            piece = PAWN;
        else if (queens & src)
            piece = QUEEN;
        else if (pos->bishopQueens & src)
            piece = BISHOP;
        else
            piece = ROOK;

        if (updateHash)
        {
            // remove moving piece from source
            hash ^= ZOB_KEY(pieces[chance][piece - 1][move.getFrom()]);
        }

        // promote the pawn (if this was promotion move)
        if (move.getFlags() == CM_FLAG_KNIGHT_PROMOTION || move.getFlags() == CM_FLAG_KNIGHT_PROMO_CAP)
            piece = KNIGHT;
        else if (move.getFlags() == CM_FLAG_BISHOP_PROMOTION || move.getFlags() == CM_FLAG_BISHOP_PROMO_CAP)
            piece = BISHOP;
        else if (move.getFlags() == CM_FLAG_ROOK_PROMOTION || move.getFlags() == CM_FLAG_ROOK_PROMO_CAP)
            piece = ROOK;
        else if (move.getFlags() == CM_FLAG_QUEEN_PROMOTION || move.getFlags() == CM_FLAG_QUEEN_PROMO_CAP)
            piece = QUEEN;

        if (updateHash)
        {
            // remove captured piece from dst
            {
                uint8 dstPiece = 0;
                // figure out destination piece
                if (pos->kings & dst)
                    dstPiece = KING;
                else if (pos->knights & dst)
                    dstPiece = KNIGHT;
                else if ((pos->pawns & RANKS2TO7) & dst)
                    dstPiece = PAWN;
                else if (queens & dst)
                    dstPiece = QUEEN;
                else if (pos->bishopQueens & dst)
                    dstPiece = BISHOP;
                else if (pos->rookQueens & dst)
                    dstPiece = ROOK;

                if (dstPiece)
                {
                    hash ^= ZOB_KEY(pieces[!chance][dstPiece - 1][move.getTo()]);
                }
            }

            // add moving piece at dst
            hash ^= ZOB_KEY(pieces[chance][piece - 1][move.getTo()]);

            // flip color
            hash ^= ZOB_KEY(chance);

            // clear special move flags
            // castling rights
            if (pos->whiteCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY(castlingRights[WHITE][0]);
            if (pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY(castlingRights[WHITE][1]);

            if (pos->blackCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY(castlingRights[BLACK][0]);
            if (pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY(castlingRights[BLACK][1]);


            // en-passent target
            if (pos->enPassent)
            {
                hash ^= ZOB_KEY(enPassentTarget[pos->enPassent - 1]);
            }
        }

        // remove source from all bitboards
        pos->bishopQueens &= ~src;
        pos->rookQueens   &= ~src;
        pos->kings        &= ~src;
        pos->knights      &= ~src;
        pos->pawns        &= ~(src & RANKS2TO7);

        // remove the dst from all bitboards
        pos->bishopQueens &= ~dst;
        pos->rookQueens   &= ~dst;
        pos->kings        &= ~dst;
        pos->knights      &= ~dst;
        pos->pawns        &= ~(dst & RANKS2TO7);

        // put the piece that moved in the required bitboards
        if (piece == KING)
        {
            pos->kings          |= dst;

            if (chance == WHITE)
                pos->whiteCastle = 0;
            else
                pos->blackCastle = 0;
        }

        if (piece == KNIGHT)
            pos->knights        |= dst;

        if (piece == PAWN)
            pos->pawns          |= dst;

        if (piece == BISHOP || piece == QUEEN)
            pos->bishopQueens   |= dst;

        if (piece == ROOK || piece == QUEEN)
            pos->rookQueens     |= dst;


        if (chance == WHITE)
        {
            pos->whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            pos->whitePieces  = pos->whitePieces  & ~dst;
        }

        // if it's an en-passet move, remove the captured pawn also
        if (move.getFlags() == CM_FLAG_EP_CAPTURE)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(dst) : northOne(dst);

            pos->pawns              &= ~(enPassentCapturedPiece & RANKS2TO7);

            if (updateHash)
            {
                hash ^= ZOB_KEY(pieces[!chance][ZOB_INDEX_PAWN][bitScan(enPassentCapturedPiece)]);
            }

            if (chance == BLACK)
                pos->whitePieces    &= ~enPassentCapturedPiece;
        }

        // if it's a castling, move the rook also
        if (chance == WHITE)
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // white castle king side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(H1)) | BIT(F1);
                pos->whitePieces = (pos->whitePieces ^ BIT(H1)) | BIT(F1);
                if (updateHash)
                {
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][H1]);
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][F1]);
                }
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // white castle queen side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(A1)) | BIT(D1);
                pos->whitePieces = (pos->whitePieces ^ BIT(A1)) | BIT(D1);
                if (updateHash)
                {
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][A1]);
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][D1]);
                }
            }
        }
        else
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // black castle king side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(H8)) | BIT(F8);
                if (updateHash)
                {
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][H8]);
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][F8]);
                }
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // black castle queen side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(A8)) | BIT(D8);
                if (updateHash)
                {
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][A8]);
                    hash ^= ZOB_KEY(pieces[chance][ZOB_INDEX_ROOK][D8]);
                }
            }
        }


        // update the game state
        pos->chance = !chance;
        pos->enPassent = 0;
        //pos->halfMoveCounter++;   // quiet move -> increment half move counter // TODO: correctly increment this based on if there was a capture
        updateCastleFlag(pos, dst,  chance);

        if (piece == ROOK)
        {
            updateCastleFlag(pos, src, !chance);
        }

        if (move.getFlags() == CM_FLAG_DOUBLE_PAWN_PUSH)
        {
#if EXACT_EN_PASSENT_FLAGGING == 1
            // only mark en-passent if there actually is a en-passent capture possible in next move
            uint64 allPawns     = pos->pawns & RANKS2TO7;    // get rid of game state variables
            uint64 allPieces    = pos->kings |  allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
            uint64 blackPieces  = allPieces & (~pos->whitePieces);
            uint64 enemyPieces  = (chance == WHITE) ? blackPieces : pos->whitePieces;
            uint64 enemyPawns   = allPawns & enemyPieces;


            // possible enemy pieces that can do en-passent capture in next move
            uint64 epSources = (eastOne(dst) | westOne(dst)) & enemyPawns;

            if (epSources)
#endif
            {
                pos->enPassent =  (move.getFrom() & 7) + 1;      // store file + 1
            }
        }

        if (updateHash)
        {
            // add special move flags
            // castling rights
            if (pos->whiteCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY(castlingRights[WHITE][0]);
            if (pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY(castlingRights[WHITE][1]);

            if (pos->blackCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY(castlingRights[BLACK][0]);
            if (pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY(castlingRights[BLACK][1]);


            // en-passent target
            if (pos->enPassent)
            {
                hash ^= ZOB_KEY(enPassentTarget[pos->enPassent - 1]);
            }
        }
    }
    // same as the above
    // templates + pre-processor hack doesn't work at the same time :-/
#if USE_TEMPLATE_CHANCE_OPT == 1
    template<uint8 chance, bool updateHash>
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void makeMove (HexaBitBoardPosition *pos, HashKey128b &hash, CMove move)
#else
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void makeMove(HexaBitBoardPosition *pos, HashKey128b &hash, CMove move, uint8 chance, bool updateHash)
#endif
    {
        uint64 src = BIT(move.getFrom());
        uint64 dst = BIT(move.getTo());

        // figure out the source piece
        uint64 queens = pos->bishopQueens & pos->rookQueens;
        uint8 piece = 0;
        if (pos->kings & src)
            piece = KING;
        else if (pos->knights & src)
            piece = KNIGHT;
        else if ((pos->pawns & RANKS2TO7) & src)
            piece = PAWN;
        else if (queens & src)
            piece = QUEEN;
        else if (pos->bishopQueens & src)
            piece = BISHOP;
        else
            piece = ROOK;

        if (updateHash)
        {
            // remove moving piece from source
            hash ^= ZOB_KEY_128(pieces[chance][piece - 1][move.getFrom()]);
        }

        // promote the pawn (if this was promotion move)
        if (move.getFlags() == CM_FLAG_KNIGHT_PROMOTION || move.getFlags() == CM_FLAG_KNIGHT_PROMO_CAP)
            piece = KNIGHT;
        else if (move.getFlags() == CM_FLAG_BISHOP_PROMOTION || move.getFlags() == CM_FLAG_BISHOP_PROMO_CAP)
            piece = BISHOP;
        else if (move.getFlags() == CM_FLAG_ROOK_PROMOTION || move.getFlags() == CM_FLAG_ROOK_PROMO_CAP)
            piece = ROOK;
        else if (move.getFlags() == CM_FLAG_QUEEN_PROMOTION || move.getFlags() == CM_FLAG_QUEEN_PROMO_CAP)
            piece = QUEEN;

        if (updateHash)
        {
            // remove captured piece from dst
            {
                uint8 dstPiece = 0;
                // figure out destination piece
                if (pos->kings & dst)
                    dstPiece = KING;
                else if (pos->knights & dst)
                    dstPiece = KNIGHT;
                else if ((pos->pawns & RANKS2TO7) & dst)
                    dstPiece = PAWN;
                else if (queens & dst)
                    dstPiece = QUEEN;
                else if (pos->bishopQueens & dst)
                    dstPiece = BISHOP;
                else if (pos->rookQueens & dst)
                    dstPiece = ROOK;

                if (dstPiece)
                {
                    hash ^= ZOB_KEY_128(pieces[!chance][dstPiece - 1][move.getTo()]);
                }
            }

            // add moving piece at dst
            hash ^= ZOB_KEY_128(pieces[chance][piece - 1][move.getTo()]);

            // flip color
            hash ^= ZOB_KEY_128(chance);

            // clear special move flags
            // castling rights
            if (pos->whiteCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[WHITE][0]);
            if (pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[WHITE][1]);

            if (pos->blackCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[BLACK][0]);
            if (pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[BLACK][1]);


            // en-passent target
            if (pos->enPassent)
            {
                hash ^= ZOB_KEY_128(enPassentTarget[pos->enPassent - 1]);
            }
        }

        // remove source from all bitboards
        pos->bishopQueens &= ~src;
        pos->rookQueens   &= ~src;
        pos->kings        &= ~src;
        pos->knights      &= ~src;
        pos->pawns        &= ~(src & RANKS2TO7);

        // remove the dst from all bitboards
        pos->bishopQueens &= ~dst;
        pos->rookQueens   &= ~dst;
        pos->kings        &= ~dst;
        pos->knights      &= ~dst;
        pos->pawns        &= ~(dst & RANKS2TO7);

        // put the piece that moved in the required bitboards
        if (piece == KING)
        {
            pos->kings          |= dst;

            if (chance == WHITE)
                pos->whiteCastle = 0;
            else
                pos->blackCastle = 0;
        }

        if (piece == KNIGHT)
            pos->knights        |= dst;

        if (piece == PAWN)
            pos->pawns          |= dst;

        if (piece == BISHOP || piece == QUEEN)
            pos->bishopQueens   |= dst;

        if (piece == ROOK || piece == QUEEN)
            pos->rookQueens     |= dst;


        if (chance == WHITE)
        {
            pos->whitePieces = (pos->whitePieces ^ src) | dst;
        }
        else
        {
            pos->whitePieces  = pos->whitePieces  & ~dst;
        }

        // if it's an en-passet move, remove the captured pawn also
        if (move.getFlags() == CM_FLAG_EP_CAPTURE)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(dst) : northOne(dst);

            pos->pawns              &= ~(enPassentCapturedPiece & RANKS2TO7);

            if (updateHash)
            {
                hash ^= ZOB_KEY_128(pieces[!chance][ZOB_INDEX_PAWN][bitScan(enPassentCapturedPiece)]);
            }

            if (chance == BLACK)
                pos->whitePieces    &= ~enPassentCapturedPiece;
        }

        // if it's a castling, move the rook also
        if (chance == WHITE)
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // white castle king side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(H1)) | BIT(F1);
                pos->whitePieces = (pos->whitePieces ^ BIT(H1)) | BIT(F1);
                if (updateHash)
                {
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][H1]);
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][F1]);
                }
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // white castle queen side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(A1)) | BIT(D1);
                pos->whitePieces = (pos->whitePieces ^ BIT(A1)) | BIT(D1);
                if (updateHash)
                {
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][A1]);
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][D1]);
                }
            }
        }
        else
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // black castle king side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(H8)) | BIT(F8);
                if (updateHash)
                {
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][H8]);
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][F8]);
                }
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // black castle queen side
                pos->rookQueens  = (pos->rookQueens  ^ BIT(A8)) | BIT(D8);
                if (updateHash)
                {
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][A8]);
                    hash ^= ZOB_KEY_128(pieces[chance][ZOB_INDEX_ROOK][D8]);
                }
            }
        }


        // update the game state
        pos->chance = !chance;
        pos->enPassent = 0;
        //pos->halfMoveCounter++;   // quiet move -> increment half move counter // TODO: correctly increment this based on if there was a capture
        updateCastleFlag(pos, dst,  chance);

        if (piece == ROOK)
        {
            updateCastleFlag(pos, src, !chance);
        }

        if (move.getFlags() == CM_FLAG_DOUBLE_PAWN_PUSH)
        {
#if EXACT_EN_PASSENT_FLAGGING == 1
            // only mark en-passent if there actually is a en-passent capture possible in next move
            uint64 allPawns     = pos->pawns & RANKS2TO7;    // get rid of game state variables
            uint64 allPieces    = pos->kings |  allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
            uint64 blackPieces  = allPieces & (~pos->whitePieces);
            uint64 enemyPieces  = (chance == WHITE) ? blackPieces : pos->whitePieces;
            uint64 enemyPawns   = allPawns & enemyPieces;


            // possible enemy pieces that can do en-passent capture in next move
            uint64 epSources = (eastOne(dst) | westOne(dst)) & enemyPawns;

            if (epSources)
#endif
            {
                pos->enPassent =  (move.getFrom() & 7) + 1;      // store file + 1
            }
        }

        if (updateHash)
        {
            // add special move flags
            // castling rights
            if (pos->whiteCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[WHITE][0]);
            if (pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[WHITE][1]);

            if (pos->blackCastle & CASTLE_FLAG_KING_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[BLACK][0]);
            if (pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
                hash ^= ZOB_KEY_128(castlingRights[BLACK][1]);


            // en-passent target
            if (pos->enPassent)
            {
                hash ^= ZOB_KEY_128(enPassentTarget[pos->enPassent - 1]);
            }
        }

    }

    // compute zobrist hash key for a given board position
    static CUDA_CALLABLE_MEMBER uint64 computeZobristKey(HexaBitBoardPosition *pos)
    {
        uint64 key = 0;

        // chance (side to move)
        if (pos->chance == WHITE)
            key ^= ZOB_KEY(chance);

        // castling rights
        if (pos->whiteCastle & CASTLE_FLAG_KING_SIDE)
            key ^= ZOB_KEY(castlingRights[WHITE][0]);
        if (pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
            key ^= ZOB_KEY(castlingRights[WHITE][1]);

        if (pos->blackCastle & CASTLE_FLAG_KING_SIDE)
            key ^= ZOB_KEY(castlingRights[BLACK][0]);
        if (pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
            key ^= ZOB_KEY(castlingRights[BLACK][1]);


        // en-passent target
        if (pos->enPassent)
        {
            key ^= ZOB_KEY(enPassentTarget[pos->enPassent - 1]);
        }
        

        // piece-position
        uint64 allPawns     = pos->pawns & RANKS2TO7;    // get rid of game state variables
        uint64 allPieces    = pos->kings |  allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;

        while(allPieces)
        {
            uint64 piece = getOne(allPieces); // FIXED: Was MoveGeneratorBitboard::getOne
            int square = bitScan(piece);

            int color = (piece & pos->whitePieces) ? WHITE : BLACK;
            if (piece & allPawns)
            {
                key ^= ZOB_KEY(pieces[color][ZOB_INDEX_PAWN][square]);
            }
            else if (piece & pos->kings)
            {
                key ^= ZOB_KEY(pieces[color][ZOB_INDEX_KING][square]);
            }
            else if (piece & pos->knights)
            {
                key ^= ZOB_KEY(pieces[color][ZOB_INDEX_KNIGHT][square]);
            }
            else if (piece & pos->rookQueens & pos->bishopQueens)
            {
                key ^= ZOB_KEY(pieces[color][ZOB_INDEX_QUEEN][square]);
            }
            else if (piece & pos->rookQueens)
            {
                key ^= ZOB_KEY(pieces[color][ZOB_INDEX_ROOK][square]);
            }
            else if (piece & pos->bishopQueens)
            {
                key ^= ZOB_KEY(pieces[color][ZOB_INDEX_BISHOP][square]);
            }

            allPieces ^= piece;
        }

        return key;
    }
    // compute zobrist hash key for a given board position (128 bit hash)
    static CUDA_CALLABLE_MEMBER HashKey128b computeZobristKey128b(HexaBitBoardPosition *pos)
    {
        HashKey128b key(0,0);

        // chance (side to move)
        if (pos->chance == WHITE)
            key ^= ZOB_KEY_128(chance);

        // castling rights
        if (pos->whiteCastle & CASTLE_FLAG_KING_SIDE)
            key ^= ZOB_KEY_128(castlingRights[WHITE][0]);
        if (pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
            key ^= ZOB_KEY_128(castlingRights[WHITE][1]);

        if (pos->blackCastle & CASTLE_FLAG_KING_SIDE)
            key ^= ZOB_KEY_128(castlingRights[BLACK][0]);
        if (pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
            key ^= ZOB_KEY_128(castlingRights[BLACK][1]);


        // en-passent target
        if (pos->enPassent)
        {
            key ^= ZOB_KEY_128(enPassentTarget[pos->enPassent - 1]);
        }


        // piece-position
        uint64 allPawns = pos->pawns & RANKS2TO7;    // get rid of game state variables
        uint64 allPieces = pos->kings | allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;

        while (allPieces)
        {
            uint64 piece = getOne(allPieces); // FIXED: Was MoveGeneratorBitboard::getOne
            int square = bitScan(piece);

            int color = (piece & pos->whitePieces) ? WHITE : BLACK;
            if (piece & allPawns)
            {
                key ^= ZOB_KEY_128(pieces[color][ZOB_INDEX_PAWN][square]);
            }
            else if (piece & pos->kings)
            {
                key ^= ZOB_KEY_128(pieces[color][ZOB_INDEX_KING][square]);
            }
            else if (piece & pos->knights)
            {
                key ^= ZOB_KEY_128(pieces[color][ZOB_INDEX_KNIGHT][square]);
            }
            else if (piece & pos->rookQueens & pos->bishopQueens)
            {
                key ^= ZOB_KEY_128(pieces[color][ZOB_INDEX_QUEEN][square]);
            }
            else if (piece & pos->rookQueens)
            {
                key ^= ZOB_KEY_128(pieces[color][ZOB_INDEX_ROOK][square]);
            }
            else if (piece & pos->bishopQueens)
            {
                key ^= ZOB_KEY_128(pieces[color][ZOB_INDEX_BISHOP][square]);
            }

            allPieces ^= piece;
        }

        return key;
    }

#endif // MOVE_MAKER_H_
