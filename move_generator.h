#ifndef MOVE_GENERATOR_H_
#define MOVE_GENERATOR_H_

#include "attack_generators.h"
#include "castling_logic.h"
#include "ep_logic.h"
#include "move_encoder.h"
#include "pawn_logic.h"
#include "bitboard_utils.h"

// This file contains the implementations of the move generation algorithms.
// It is included by MoveGeneratorBitboard.cu to build the final functions.

static void initialize_move_generator_tables()
{
    memcpy(&zob, &randoms[1200], sizeof(zob));
    memcpy(&zob2, &randoms[333], sizeof(zob2));

    for (uint8 i=0; i < 64; i++) {
        uint64 x = BIT(i);
        uint64 north = northAttacks(x, ALLSET);
        uint64 south = southAttacks(x, ALLSET);
        uint64 east  = eastAttacks (x, ALLSET);
        uint64 west  = westAttacks (x, ALLSET);
        uint64 ne    = northEastAttacks(x, ALLSET);
        uint64 nw    = northWestAttacks(x, ALLSET);
        uint64 se    = southEastAttacks(x, ALLSET);
        uint64 sw    = southWestAttacks(x, ALLSET);

        RookAttacks  [i] = north | south | east | west;
        BishopAttacks[i] = ne | nw | se | sw;
        QueenAttacks [i] = RookAttacks[i] | BishopAttacks[i];
        KnightAttacks[i] = knightAttacks(x);
        KingAttacks[i]   = kingAttacks(x);
    }

    for (uint8 i=0; i<64; i++) {
        for (uint8 j=0; j<64; j++) {
            if (i <= j) {
                Between[i][j] = squaresInBetween(i, j);
                Between[j][i] = Between[i][j];
            }
            Line[i][j] = squaresInLine(i, j);
        }
    }

#if USE_SLIDING_LUT == 1
    srand (time(NULL));
    for (int square = A1; square <= H8; square++) {
        uint64 thisSquare = BIT(square);
        uint64 mask = sqRookAttacks(square) & (~thisSquare);
        if ((thisSquare & RANK1) == 0) mask &= ~RANK1;
        if ((thisSquare & RANK8) == 0) mask &= ~RANK8;
        if ((thisSquare & FILEA) == 0) mask &= ~FILEA;
        if ((thisSquare & FILEH) == 0) mask &= ~FILEH;
        RookAttacksMasked[square] = mask;
        mask = sqBishopAttacks(square)  & (~thisSquare) & CENTRAL_SQUARES;
        BishopAttacksMasked[square] = mask;
    }
    memset(fancy_magic_lookup_table, 0, sizeof(fancy_magic_lookup_table));
    int globalOffsetRook = 0;
    int globalOffsetBishop = 0;
    for (int square = A1; square <= H8; square++) {
        int uniqueBishopAttacks = 0, uniqueRookAttacks=0;
        uint64 rookMagic = findRookMagicForSquare(square, &fancy_magic_lookup_table[rook_magics_fancy[square].position], rook_magics_fancy[square].factor, &fancy_byte_RookLookup[globalOffsetRook], &fancy_byte_magic_lookup_table[rook_magics_fancy[square].position], &uniqueRookAttacks);
        assert(rookMagic == rook_magics_fancy[square].factor);
        uint64 bishopMagic = findBishopMagicForSquare(square, &fancy_magic_lookup_table[bishop_magics_fancy[square].position], bishop_magics_fancy[square].factor, &fancy_byte_BishopLookup[globalOffsetBishop], &fancy_byte_magic_lookup_table[bishop_magics_fancy[square].position], &uniqueBishopAttacks);
        assert(bishopMagic == bishop_magics_fancy[square].factor);
        rook_magics_fancy[square].offset = globalOffsetRook;
        globalOffsetRook += uniqueRookAttacks;
        bishop_magics_fancy[square].offset = globalOffsetBishop;
        globalOffsetBishop += uniqueBishopAttacks;
    }
#endif

#ifndef SKIP_CUDA_CODE
    cudaMemcpyToSymbol(gBetween, Between, sizeof(Between));
    cudaMemcpyToSymbol(gLine, Line, sizeof(Line));
    cudaMemcpyToSymbol(gRookAttacks, RookAttacks, sizeof(RookAttacks));
    cudaMemcpyToSymbol(gBishopAttacks, BishopAttacks, sizeof(BishopAttacks));
    cudaMemcpyToSymbol(gQueenAttacks, QueenAttacks, sizeof(QueenAttacks));
    cudaMemcpyToSymbol(gKnightAttacks, KnightAttacks, sizeof(KnightAttacks));
    cudaMemcpyToSymbol(gKingAttacks, KingAttacks, sizeof(KingAttacks));
    cudaMemcpyToSymbol(gRookAttacksMasked, RookAttacksMasked, sizeof(RookAttacksMasked));
    cudaMemcpyToSymbol(gBishopAttacksMasked, BishopAttacksMasked , sizeof(BishopAttacksMasked));
    cudaMemcpyToSymbol(gRookMagics, rookMagics, sizeof(rookMagics));
    cudaMemcpyToSymbol(gBishopMagics, bishopMagics, sizeof(bishopMagics));
    cudaMemcpyToSymbol(gRookMagicAttackTables, rookMagicAttackTables, sizeof(rookMagicAttackTables));
    cudaMemcpyToSymbol(gBishopMagicAttackTables, bishopMagicAttackTables, sizeof(bishopMagicAttackTables));
    cudaMemcpyToSymbol(g_fancy_magic_lookup_table, fancy_magic_lookup_table, sizeof(fancy_magic_lookup_table));
    cudaMemcpyToSymbol(g_bishop_magics_fancy, bishop_magics_fancy, sizeof(bishop_magics_fancy));
    cudaMemcpyToSymbol(g_rook_magics_fancy, rook_magics_fancy, sizeof(rook_magics_fancy));
    cudaMemcpyToSymbol(g_fancy_byte_magic_lookup_table, fancy_byte_magic_lookup_table, sizeof(fancy_byte_magic_lookup_table));
    cudaMemcpyToSymbol(g_fancy_byte_BishopLookup, fancy_byte_BishopLookup, sizeof(fancy_byte_BishopLookup));
    cudaMemcpyToSymbol(g_fancy_byte_RookLookup, fancy_byte_RookLookup, sizeof(fancy_byte_RookLookup));
    
    void *tempPtr = NULL;
    cudaGetSymbolAddress(&tempPtr, gZob);
    cudaMemcpy(tempPtr, &zob, sizeof(zob), cudaMemcpyHostToDevice);
    cudaGetSymbolAddress(&tempPtr, gZob2);
    cudaMemcpy(tempPtr, &zob2, sizeof(zob2), cudaMemcpyHostToDevice);
#endif
}

// ============================================================================
// FULL MOVE GENERATION IMPLEMENTATIONS
// ============================================================================

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 findPinnedPieces_internal (uint64 myKing, uint64 myPieces, uint64 enemyBishops, uint64 enemyRooks, uint64 allPieces, uint8 kingIndex)
{
    uint64 b = sqBishopAttacks(kingIndex) & enemyBishops;
    uint64 r = sqRookAttacks  (kingIndex) & enemyRooks;
    uint64 attackers = b | r;
    uint64 pinned = EMPTY;
    while (attackers) {
        uint64 attacker = getOne(attackers);
        uint8 attackerIndex = bitScan(attacker);
        uint64 squaresInBetween = sqsInBetween(attackerIndex, kingIndex);
        uint64 piecesInBetween = squaresInBetween & allPieces;
        if (isSingular(piecesInBetween)) pinned |= piecesInBetween;
        attackers ^= attacker;
    }
    return pinned;
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 findAttackedSquares_internal(uint64 emptySquares, uint64 enemyBishops, uint64 enemyRooks, uint64 enemyPawns, uint64 enemyKnights, uint64 enemyKing, uint64 myKing, uint8 enemyColor)
{
    uint64 attacked = 0;
    if (enemyColor == WHITE) {
        attacked |= northEastOne(enemyPawns);
        attacked |= northWestOne(enemyPawns);
    } else {
        attacked |= southEastOne(enemyPawns);
        attacked |= southWestOne(enemyPawns);
    }
    attacked |= knightAttacks(enemyKnights);
    attacked |= multiBishopAttacks(enemyBishops, emptySquares | myKing);
    attacked |= multiRookAttacks(enemyRooks, emptySquares | myKing);
    attacked |= kingAttacks(enemyKing);
    return attacked;
}

#if USE_TEMPLATE_CHANCE_OPT == 1
template<uint8 chance>
#endif
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint32 generateBoardsOutOfCheck_internal (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions, uint64 allPawns, uint64 allPieces, uint64 myPieces, uint64 enemyPieces, uint64 pinned, uint64 threatened, uint8 kingIndex
#if USE_TEMPLATE_CHANCE_OPT != 1
                                       , uint8 chance
#endif
                                       )
{
    uint32 nMoves = 0;
    uint64 king = pos->kings & myPieces;
    uint64 attackers = 0;
    uint64 enemyPawns = allPawns & enemyPieces;
    attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) : (southEastOne(king) | southWestOne(king)) ) & enemyPawns;
    uint64 enemyKnights = pos->knights & enemyPieces;
    attackers |= knightAttacks(king) & enemyKnights;
    uint64 enemyBishops = pos->bishopQueens & enemyPieces;
    attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;
    uint64 enemyRooks = pos->rookQueens & enemyPieces;
    attackers |= rookAttacks(king, ~allPieces) & enemyRooks;
    uint64 kingMoves = kingAttacks(king);
    kingMoves &= ~(threatened | myPieces);
    while(kingMoves) {
        uint64 dst = getOne(kingMoves);
        addKingMove_internal(&nMoves, &newPositions, pos, king, dst, chance);
        kingMoves ^= dst;
    }
    if (isSingular(attackers)) {
        uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));
        myPieces &= ~pinned;
        uint64 myPawns = allPawns & myPieces;
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);
        uint64 enPassentTarget = 0;
        if (pos->enPassent) {
            if (chance == BLACK) enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
            else enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
        }
        uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
        if (enPassentCapturedPiece != attackers) enPassentTarget = 0;
        while (myPawns) {
            uint64 pawn = getOne(myPawns);
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
            if (dst) {
                if (dst & safeSquares) {
                    addPawnMoves_internal(&nMoves, &newPositions, pos, pawn, dst, chance);
                } else {
                    dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);
                    if (dst) addSinglePawnMove_internal(&nMoves, &newPositions, pos, pawn, dst, chance, true, bitScan(pawn));
                }
            }
            uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
            uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
            dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
            if (dst) addPawnMoves_internal(&nMoves, &newPositions, pos, pawn, dst, chance);
            dst = (westCapture | eastCapture) & enPassentTarget;
            if (dst) addEnPassentMove_internal(&nMoves, &newPositions, pos, pawn, dst, chance);
            myPawns ^= pawn;
        }
        uint64 myKnights = (pos->knights & myPieces);
        while (myKnights) {
            uint64 knight = getOne(myKnights);
            uint64 knightMoves = knightAttacks(knight) & safeSquares;
            while (knightMoves) {
                uint64 dst = getOne(knightMoves);
                addKnightMove_internal(&nMoves, &newPositions, pos, knight, dst, chance);
                knightMoves ^= dst;
            }
            myKnights ^= knight;
        }
        uint64 bishops = pos->bishopQueens & myPieces;
        while (bishops) {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;
            while (bishopMoves) {
                uint64 dst = getOne(bishopMoves);
                addSlidingMove_internal(&nMoves, &newPositions, pos, bishop, dst, chance);
                bishopMoves ^= dst;
            }
            bishops ^= bishop;
        }
        uint64 rooks = pos->rookQueens & myPieces;
        while (rooks) {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;
            while (rookMoves) {
                uint64 dst = getOne(rookMoves);
                addSlidingMove_internal(&nMoves, &newPositions, pos, rook, dst, chance);
                rookMoves ^= dst;
            }
            rooks ^= rook;
        }
    }
    return nMoves;
}

#if USE_TEMPLATE_CHANCE_OPT == 1
template <uint8 chance>
CUDA_CALLABLE_MEMBER static uint32 generateBoards_internal (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions)
#else
CUDA_CALLABLE_MEMBER static uint32 generateBoards_internal (HexaBitBoardPosition *pos, HexaBitBoardPosition *newPositions, uint8 chance)
#endif
{
    uint32 nMoves = 0;
    uint64 allPawns = pos->pawns & RANKS2TO7;
    uint64 allPieces = pos->kings | allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
    uint64 blackPieces = allPieces & (~pos->whitePieces);
    uint64 myPieces = (chance == WHITE) ? pos->whitePieces : blackPieces;
    uint64 enemyPieces = (chance == WHITE) ? blackPieces : pos->whitePieces;
    uint64 enemyBishops = pos->bishopQueens & enemyPieces;
    uint64 enemyRooks = pos->rookQueens & enemyPieces;
    uint64 myKing = pos->kings & myPieces;
    uint8 kingIndex = bitScan(myKing);
    uint64 pinned = findPinnedPieces_internal(myKing, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);
    uint64 threatened = findAttackedSquares_internal(~allPieces, enemyBishops, enemyRooks, allPawns & enemyPieces, pos->knights & enemyPieces, pos->kings & enemyPieces, myKing, !chance);

    if (threatened & myKing) {
#if USE_TEMPLATE_CHANCE_OPT == 1
        return generateBoardsOutOfCheck_internal<chance>(pos, newPositions, allPawns, allPieces, myPieces, enemyPieces, pinned, threatened, kingIndex);
#else
        return generateBoardsOutOfCheck_internal (pos, newPositions, allPawns, allPieces, myPieces, enemyPieces, pinned, threatened, kingIndex, chance);
#endif
    }

    uint64 myPawns = allPawns & myPieces;
    uint64 enPassentTarget = 0;
    if (pos->enPassent) {
        if (chance == BLACK) enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
        else enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
    }
    if (enPassentTarget) {
        uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
        uint64 epSources = (eastOne(enPassentCapturedPiece) | westOne(enPassentCapturedPiece)) & myPawns;
        while (epSources) {
            uint64 pawn = getOne(epSources);
            if (pawn & pinned) {
                uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                if (enPassentTarget & line) addEnPassentMove_internal(&nMoves, &newPositions, pos, pawn, enPassentTarget, chance);
            } else {
                uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & myKing;
                if (!causesCheck) addEnPassentMove_internal(&nMoves, &newPositions, pos, pawn, enPassentTarget, chance);
            }
            epSources ^= pawn;
        }
    }

    uint64 checkingRankDoublePush = RANK3 << (chance * 24);
    uint64 pinnedPawns = myPawns & pinned;
    while (pinnedPawns) {
        uint64 pawn = getOne(pinnedPawns);
        uint8 pawnIndex = bitScan(pawn);
        uint64 line = sqsInLine(pawnIndex, kingIndex);
        uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
        if (dst) {
            addSinglePawnMove_internal(&nMoves, &newPositions, pos, pawn, dst, chance, false, pawnIndex);
            dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (~allPieces);
            if (dst) addSinglePawnMove_internal(&nMoves, &newPositions, pos, pawn, dst, chance, true, pawnIndex);
        }
        dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
        dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
        if (dst & enemyPieces) addPawnMoves_internal(&nMoves, &newPositions, pos, pawn, dst, chance);
        pinnedPawns ^= pawn;
    }

    myPawns &= ~pinned;
    while (myPawns) {
        uint64 pawn = getOne(myPawns);
        uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
        if (dst) {
            addPawnMoves_internal(&nMoves, &newPositions, pos, pawn, dst, chance);
            dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (~allPieces);
            if (dst) addSinglePawnMove_internal(&nMoves, &newPositions, pos, pawn, dst, chance, true, bitScan(pawn));
        }
        uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
        dst = westCapture & enemyPieces;
        if (dst) addPawnMoves_internal(&nMoves, &newPositions, pos, pawn, dst, chance);
        uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
        dst = eastCapture & enemyPieces;
        if (dst) addPawnMoves_internal(&nMoves, &newPositions, pos, pawn, dst, chance);
        myPawns ^= pawn;
    }

    if (chance == WHITE) {
        if ((pos->whiteCastle & CASTLE_FLAG_KING_SIDE) && !(F1G1 & allPieces) && !(F1G1 & threatened))
            addCastleMove_internal(&nMoves, &newPositions, pos, BIT(E1), BIT(G1), BIT(H1), BIT(F1), chance);
        if ((pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) && !(B1D1 & allPieces) && !(C1D1 & threatened))
            addCastleMove_internal(&nMoves, &newPositions, pos, BIT(E1), BIT(C1), BIT(A1), BIT(D1), chance);
    } else {
        if ((pos->blackCastle & CASTLE_FLAG_KING_SIDE) && !(F8G8 & allPieces) && !(F8G8 & threatened))
            addCastleMove_internal(&nMoves, &newPositions, pos, BIT(E8), BIT(G8), BIT(H8), BIT(F8), chance);
        if ((pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE) && !(B8D8 & allPieces) && !(C8D8 & threatened))
            addCastleMove_internal(&nMoves, &newPositions, pos, BIT(E8), BIT(C8), BIT(A8), BIT(D8), chance);
    }
    
    uint64 kingMoves = kingAttacks(myKing);
    kingMoves &= ~(threatened | myPieces);
    while(kingMoves) {
        uint64 dst = getOne(kingMoves);
        addKingMove_internal(&nMoves, &newPositions, pos, myKing, dst, chance);
        kingMoves ^= dst;
    }

    uint64 myKnights = (pos->knights & myPieces) & ~pinned;
    while (myKnights) {
        uint64 knight = getOne(myKnights);
        uint64 knightMoves = knightAttacks(knight) & ~myPieces;
        while (knightMoves) {
            uint64 dst = getOne(knightMoves);
            addKnightMove_internal(&nMoves, &newPositions, pos, knight, dst, chance);
            knightMoves ^= dst;
        }
        myKnights ^= knight;
    }

    uint64 myBishops = pos->bishopQueens & myPieces;
    uint64 bishops = myBishops & pinned;
    while (bishops) {
        uint64 bishop = getOne(bishops);
        uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
        bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);
        while (bishopMoves) {
            uint64 dst = getOne(bishopMoves);
            addSlidingMove_internal(&nMoves, &newPositions, pos, bishop, dst, chance);
            bishopMoves ^= dst;
        }
        bishops ^= bishop;
    }
    bishops = myBishops & ~pinned;
    while (bishops) {
        uint64 bishop = getOne(bishops);
        uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
        while (bishopMoves) {
            uint64 dst = getOne(bishopMoves);
            addSlidingMove_internal(&nMoves, &newPositions, pos, bishop, dst, chance);
            bishopMoves ^= dst;
        }
        bishops ^= bishop;
    }

    uint64 myRooks = pos->rookQueens & myPieces;
    uint64 rooks = myRooks & pinned;
    while (rooks) {
        uint64 rook = getOne(rooks);
        uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
        rookMoves &= sqsInLine(bitScan(rook), kingIndex);
        while (rookMoves) {
            uint64 dst = getOne(rookMoves);
            addSlidingMove_internal(&nMoves, &newPositions, pos, rook, dst, chance);
            rookMoves ^= dst;
        }
        rooks ^= rook;
    }
    rooks = myRooks & ~pinned;
    while (rooks) {
        uint64 rook = getOne(rooks);
        uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
        while (rookMoves) {
            uint64 dst = getOne(rookMoves);
            addSlidingMove_internal(&nMoves, &newPositions, pos, rook, dst, chance);
            rookMoves ^= dst;
        }
        rooks ^= rook;
    }
    return nMoves;
}

#if USE_TEMPLATE_CHANCE_OPT == 1
template<uint8 chance>
#endif
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint32 generateMovesOutOfCheck_internal (HexaBitBoardPosition *pos, CMove *genMoves, uint64 allPawns, uint64 allPieces, uint64 myPieces, uint64 enemyPieces, uint64 pinned, uint64 threatened, uint8 kingIndex
#if USE_TEMPLATE_CHANCE_OPT != 1
                                           , uint8 chance
#endif
                                           )
{
    uint32 nMoves = 0;
    uint64 king = pos->kings & myPieces;
    uint64 attackers = 0;
    uint64 enemyPawns = allPawns & enemyPieces;
    attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) : (southEastOne(king) | southWestOne(king)) ) & enemyPawns;
    uint64 enemyKnights = pos->knights & enemyPieces;
    attackers |= knightAttacks(king) & enemyKnights;
    uint64 enemyBishops = pos->bishopQueens & enemyPieces;
    attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;
    uint64 enemyRooks = pos->rookQueens & enemyPieces;
    attackers |= rookAttacks(king, ~allPieces) & enemyRooks;
    uint64 kingMoves = kingAttacks(king);
    kingMoves &= ~(threatened | myPieces);
    while(kingMoves) {
        uint64 dst = getOne(kingMoves);
        uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
        addCompactMove(&nMoves, &genMoves, kingIndex, bitScan(dst), captureFlag);
        kingMoves ^= dst;
    }
    if (isSingular(attackers)) {
        uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));
        myPieces &= ~pinned;
        uint64 myPawns = allPawns & myPieces;
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);
        uint64 enPassentTarget = 0;
        if (pos->enPassent) {
            if (chance == BLACK) enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
            else enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
        }
        uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
        if (enPassentCapturedPiece != attackers) enPassentTarget = 0;
        while (myPawns) {
            uint64 pawn = getOne(myPawns);
            uint8 from = bitScan(pawn);
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
            if (dst) {
                if (dst & safeSquares) {
                    addCompactPawnMoves(&nMoves, &genMoves, from, dst, 0);
                } else {
                    dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);
                    if (dst) addCompactMove(&nMoves, &genMoves, from, bitScan(dst), CM_FLAG_DOUBLE_PAWN_PUSH);
                }
            }
            uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
            uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
            dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
            if (dst) addCompactPawnMoves(&nMoves, &genMoves, from, dst, CM_FLAG_CAPTURE);
            dst = (westCapture | eastCapture) & enPassentTarget;
            if (dst) addCompactMove(&nMoves, &genMoves, from, bitScan(dst), CM_FLAG_EP_CAPTURE);
            myPawns ^= pawn;
        }
        uint64 myKnights = (pos->knights & myPieces);
        while (myKnights) {
            uint64 knight = getOne(myKnights);
            uint64 knightMoves = knightAttacks(knight) & safeSquares;
            while (knightMoves) {
                uint64 dst = getOne(knightMoves);
                uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
                addCompactMove(&nMoves, &genMoves, bitScan(knight), bitScan(dst), captureFlag);
                knightMoves ^= dst;
            }
            myKnights ^= knight;
        }
        uint64 bishops = pos->bishopQueens & myPieces;
        while (bishops) {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;
            while (bishopMoves) {
                uint64 dst = getOne(bishopMoves);
                uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
                addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), captureFlag);
                bishopMoves ^= dst;
            }
            bishops ^= bishop;
        }
        uint64 rooks = pos->rookQueens & myPieces;
        while (rooks) {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;
            while (rookMoves) {
                uint64 dst = getOne(rookMoves);
                uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
                addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), captureFlag);
                rookMoves ^= dst;
            }
            rooks ^= rook;
        }
    }
    return nMoves;
}

#if USE_TEMPLATE_CHANCE_OPT == 1
template <uint8 chance>
CUDA_CALLABLE_MEMBER static uint32 generateMoves_internal (HexaBitBoardPosition *pos, CMove *genMoves)
#else
CUDA_CALLABLE_MEMBER static uint32 generateMoves_internal (HexaBitBoardPosition *pos, CMove *genMoves, uint8 chance)
#endif
{
    uint32 nMoves = 0;
    uint64 allPawns = pos->pawns & RANKS2TO7;
    uint64 allPieces = pos->kings | allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
    uint64 blackPieces = allPieces & (~pos->whitePieces);
    uint64 myPieces = (chance == WHITE) ? pos->whitePieces : blackPieces;
    uint64 enemyPieces = (chance == WHITE) ? blackPieces : pos->whitePieces;
    uint64 enemyBishops = pos->bishopQueens & enemyPieces;
    uint64 enemyRooks = pos->rookQueens & enemyPieces;
    uint64 myKing = pos->kings & myPieces;
    uint8 kingIndex = bitScan(myKing);
    uint64 pinned = findPinnedPieces_internal(myKing, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);
    uint64 threatened = findAttackedSquares_internal(~allPieces, enemyBishops, enemyRooks, allPawns & enemyPieces, pos->knights & enemyPieces, pos->kings & enemyPieces, myKing, !chance);

    if (threatened & myKing) {
#if USE_TEMPLATE_CHANCE_OPT == 1
        return generateMovesOutOfCheck_internal<chance>(pos, genMoves, allPawns, allPieces, myPieces, enemyPieces, pinned, threatened, kingIndex);
#else
        return generateMovesOutOfCheck_internal (pos, genMoves, allPawns, allPieces, myPieces, enemyPieces, pinned, threatened, kingIndex, chance);
#endif
    }

    uint64 kingMoves = kingAttacks(myKing);
    kingMoves &= ~(threatened | myPieces);
    while (kingMoves) {
        uint64 dst = getOne(kingMoves);
        uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
        addCompactMove(&nMoves, &genMoves, kingIndex, bitScan(dst), captureFlag);
        kingMoves ^= dst;
    }

    uint64 myKnights = (pos->knights & myPieces) & ~pinned;
    while (myKnights) {
        uint64 knight = getOne(myKnights);
        uint64 knightMoves = knightAttacks(knight) & ~myPieces;
        while (knightMoves) {
            uint64 dst = getOne(knightMoves);
            uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
            addCompactMove(&nMoves, &genMoves, bitScan(knight), bitScan(dst), captureFlag);
            knightMoves ^= dst;
        }
        myKnights ^= knight;
    }

    uint64 myBishops = pos->bishopQueens & myPieces;
    uint64 bishops = myBishops & pinned;
    while (bishops) {
        uint64 bishop = getOne(bishops);
        uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
        bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);
        while (bishopMoves) {
            uint64 dst = getOne(bishopMoves);
            uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
            addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), captureFlag);
            bishopMoves ^= dst;
        }
        bishops ^= bishop;
    }
    bishops = myBishops & ~pinned;
    while (bishops) {
        uint64 bishop = getOne(bishops);
        uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
        while (bishopMoves) {
            uint64 dst = getOne(bishopMoves);
            uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
            addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), captureFlag);
            bishopMoves ^= dst;
        }
        bishops ^= bishop;
    }

    uint64 myRooks = pos->rookQueens & myPieces;
    uint64 rooks = myRooks & pinned;
    while (rooks) {
        uint64 rook = getOne(rooks);
        uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
        rookMoves &= sqsInLine(bitScan(rook), kingIndex);
        while (rookMoves) {
            uint64 dst = getOne(rookMoves);
            uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
            addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), captureFlag);
            rookMoves ^= dst;
        }
        rooks ^= rook;
    }
    rooks = myRooks & ~pinned;
    while (rooks) {
        uint64 rook = getOne(rooks);
        uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
        while (rookMoves) {
            uint64 dst = getOne(rookMoves);
            uint8 captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
            addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), captureFlag);
            rookMoves ^= dst;
        }
        rooks ^= rook;
    }

    uint64 myPawns = allPawns & myPieces;
    uint64 enPassentTarget = 0;
    if (pos->enPassent) {
        if (chance == BLACK) enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
        else enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
    }
    if (enPassentTarget) {
        uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
        uint64 epSources = (eastOne(enPassentCapturedPiece) | westOne(enPassentCapturedPiece)) & myPawns;
        while (epSources) {
            uint64 pawn = getOne(epSources);
            if (pawn & pinned) {
                uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                if (enPassentTarget & line) addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(enPassentTarget), CM_FLAG_EP_CAPTURE);
            } else {
                uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & myKing;
                if (!causesCheck) addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(enPassentTarget), CM_FLAG_EP_CAPTURE);
            }
            epSources ^= pawn;
        }
    }

    uint64 checkingRankDoublePush = RANK3 << (chance * 24);
    uint64 pinnedPawns = myPawns & pinned;
    while (pinnedPawns) {
        uint64 pawn = getOne(pinnedPawns);
        uint8 pawnIndex = bitScan(pawn);
        uint64 line = sqsInLine(pawnIndex, kingIndex);
        uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
        if (dst) {
            addCompactMove(&nMoves, &genMoves, pawnIndex, bitScan(dst), 0);
            dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (~allPieces);
            if (dst) addCompactMove(&nMoves, &genMoves, pawnIndex, bitScan(dst), CM_FLAG_DOUBLE_PAWN_PUSH);
        }
        dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
        dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
        if (dst & enemyPieces) addCompactPawnMoves(&nMoves, &genMoves, pawnIndex, dst, CM_FLAG_CAPTURE);
        pinnedPawns ^= pawn;
    }

    myPawns &= ~pinned;
    while (myPawns) {
        uint64 pawn = getOne(myPawns);
        uint8 from = bitScan(pawn);
        uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
        if (dst) {
            addCompactPawnMoves(&nMoves, &genMoves, from, dst, 0);
            dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (~allPieces);
            if (dst) addCompactMove(&nMoves, &genMoves, from, bitScan(dst), CM_FLAG_DOUBLE_PAWN_PUSH);
        }
        uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
        dst = westCapture & enemyPieces;
        if (dst) addCompactPawnMoves(&nMoves, &genMoves, from, dst, CM_FLAG_CAPTURE);
        uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
        dst = eastCapture & enemyPieces;
        if (dst) addCompactPawnMoves(&nMoves, &genMoves, from, dst, CM_FLAG_CAPTURE);
        myPawns ^= pawn;
    }

    if (chance == WHITE) {
        if ((pos->whiteCastle & CASTLE_FLAG_KING_SIDE) && !(F1G1 & allPieces) && !(F1G1 & threatened))
            addCompactMove(&nMoves, &genMoves, E1, G1, CM_FLAG_KING_CASTLE);
        if ((pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) && !(B1D1 & allPieces) && !(C1D1 & threatened))
            addCompactMove(&nMoves, &genMoves, E1, C1, CM_FLAG_QUEEN_CASTLE);
    } else {
        if ((pos->blackCastle & CASTLE_FLAG_KING_SIDE) && !(F8G8 & allPieces) && !(F8G8 & threatened))
            addCompactMove(&nMoves, &genMoves, E8, G8, CM_FLAG_KING_CASTLE);
        if ((pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE) && !(B8D8 & allPieces) && !(C8D8 & threatened))
            addCompactMove(&nMoves, &genMoves, E8, C8, CM_FLAG_QUEEN_CASTLE);
    }
    return nMoves;
}

#if USE_TEMPLATE_CHANCE_OPT == 1
template<uint8 chance>
#endif
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint32 countMovesOutOfCheck_internal (HexaBitBoardPosition *pos, uint64 allPawns, uint64 allPieces, uint64 myPieces, uint64 enemyPieces, uint64 pinned, uint64 threatened, uint8 kingIndex
#if USE_TEMPLATE_CHANCE_OPT != 1
                                           , uint8 chance
#endif
                                           )
{
    uint32 nMoves = 0;
    uint64 king = pos->kings & myPieces;
    uint64 attackers = 0;
    uint64 enemyPawns = allPawns & enemyPieces;
    attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) : (southEastOne(king) | southWestOne(king)) ) & enemyPawns;
    uint64 enemyKnights = pos->knights & enemyPieces;
    attackers |= knightAttacks(king) & enemyKnights;
    uint64 enemyBishops = pos->bishopQueens & enemyPieces;
    attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;
    uint64 enemyRooks = pos->rookQueens & enemyPieces;
    attackers |= rookAttacks(king, ~allPieces) & enemyRooks;
    uint64 kingMoves = kingAttacks(king);
    kingMoves &= ~(threatened | myPieces);
    nMoves += popCount(kingMoves);
    if (isSingular(attackers)) {
        uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));
        myPieces &= ~pinned;
        uint64 myPawns = allPawns & myPieces;
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);
        uint64 enPassentTarget = 0;
        if (pos->enPassent) {
            if (chance == BLACK) enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
            else enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
        }
        uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
        if (enPassentCapturedPiece != attackers) enPassentTarget = 0;
        while (myPawns) {
            uint64 pawn = getOne(myPawns);
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
            if (dst) {
                if (dst & safeSquares) {
                    if (dst & (RANK1 | RANK8)) nMoves += 4;
                    else nMoves++;
                } else {
                    dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);
                    if (dst) nMoves++;
                }
            }
            uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
            uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
            dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
            if (dst) {
                if (dst & (RANK1 | RANK8)) nMoves += 4;
                else nMoves++;
            }
            dst = (westCapture | eastCapture) & enPassentTarget;
            if (dst) nMoves++;
            myPawns ^= pawn;
        }
        uint64 myKnights = (pos->knights & myPieces);
        while (myKnights) {
            uint64 knight = getOne(myKnights);
            uint64 knightMoves = knightAttacks(knight) & safeSquares;
            nMoves += popCount(knightMoves);
            myKnights ^= knight;
        }
        uint64 bishops = pos->bishopQueens & myPieces;
        while (bishops) {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;
            nMoves += popCount(bishopMoves);
            bishops ^= bishop;
        }
        uint64 rooks = pos->rookQueens & myPieces;
        while (rooks) {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;
            nMoves += popCount(rookMoves);
            rooks ^= rook;
        }
    }
    return nMoves;
}

#if USE_TEMPLATE_CHANCE_OPT == 1
template <uint8 chance>
CUDA_CALLABLE_MEMBER static uint32 countMoves_internal (HexaBitBoardPosition *pos)
#else
CUDA_CALLABLE_MEMBER static uint32 countMoves_internal (HexaBitBoardPosition *pos, uint8 chance)
#endif
{
    uint32 nMoves = 0;
    uint64 allPawns = pos->pawns & RANKS2TO7;
    uint64 allPieces = pos->kings | allPawns | pos->knights | pos->bishopQueens | pos->rookQueens;
    uint64 blackPieces = allPieces & (~pos->whitePieces);
    uint64 myPieces = (chance == WHITE) ? pos->whitePieces : blackPieces;
    uint64 enemyPieces = (chance == WHITE) ? blackPieces : pos->whitePieces;
    uint64 enemyBishops = pos->bishopQueens & enemyPieces;
    uint64 enemyRooks = pos->rookQueens & enemyPieces;
    uint64 myKing = pos->kings & myPieces;
    uint8 kingIndex = bitScan(myKing);
    uint64 pinned = findPinnedPieces_internal(myKing, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);
    uint64 threatened = findAttackedSquares_internal(~allPieces, enemyBishops, enemyRooks, allPawns & enemyPieces, pos->knights & enemyPieces, pos->kings & enemyPieces, myKing, !chance);

    if (threatened & myKing) {
#if USE_TEMPLATE_CHANCE_OPT == 1
        return countMovesOutOfCheck_internal<chance>(pos, allPawns, allPieces, myPieces, enemyPieces, pinned, threatened, kingIndex);
#else
        return countMovesOutOfCheck_internal (pos, allPawns, allPieces, myPieces, enemyPieces, pinned, threatened, kingIndex, chance);
#endif
    }

    uint64 myPawns = allPawns & myPieces;
    uint64 enPassentTarget = 0;
    if (pos->enPassent) {
        if (chance == BLACK) enPassentTarget = BIT(pos->enPassent - 1) << (8 * 2);
        else enPassentTarget = BIT(pos->enPassent - 1) << (8 * 5);
    }
    if (enPassentTarget) {
        uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
        uint64 epSources = (eastOne(enPassentCapturedPiece) | westOne(enPassentCapturedPiece)) & myPawns;
        while (epSources) {
            uint64 pawn = getOne(epSources);
            if (pawn & pinned) {
                uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                if (enPassentTarget & line) nMoves++;
            } else {
                uint64 propogator = (~allPieces) | enPassentCapturedPiece | pawn;
                uint64 causesCheck = (eastAttacks(enemyRooks, propogator) | westAttacks(enemyRooks, propogator)) & myKing;
                if (!causesCheck) nMoves++;
            }
            epSources ^= pawn;
        }
    }

    uint64 checkingRankDoublePush = RANK3 << (chance * 24);
    uint64 pinnedPawns = myPawns & pinned;
    while (pinnedPawns) {
        uint64 pawn = getOne(pinnedPawns);
        uint8 pawnIndex = bitScan(pawn);
        uint64 line = sqsInLine(pawnIndex, kingIndex);
        uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
        if (dst) {
            nMoves++;
            dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): southOne(dst & checkingRankDoublePush) ) & (~allPieces);
            if (dst) nMoves++;
        }
        dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
        dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
        if (dst & enemyPieces) {
            if (dst & (RANK1 | RANK8)) nMoves += 4;
            else nMoves++;
        }
        pinnedPawns ^= pawn;
    }

    myPawns &= ~pinned;
    uint64 dsts = ((chance == WHITE) ? northOne(myPawns) : southOne(myPawns)) & (~allPieces);
    nMoves += popCount(dsts);
    uint64 promotions = dsts & (RANK1 | RANK8);
    nMoves += 3 * popCount(promotions);
    dsts = ((chance == WHITE) ? northOne(dsts & checkingRankDoublePush): southOne(dsts & checkingRankDoublePush) ) & (~allPieces);
    nMoves += popCount(dsts);
    dsts = ((chance == WHITE) ? northWestOne(myPawns) : southWestOne(myPawns)) & enemyPieces;
    nMoves += popCount(dsts);
    promotions = dsts & (RANK1 | RANK8);
    nMoves += 3 * popCount(promotions);
    dsts = ((chance == WHITE) ? northEastOne(myPawns) : southEastOne(myPawns)) & enemyPieces;
    nMoves += popCount(dsts);
    promotions = dsts & (RANK1 | RANK8);
    nMoves += 3 * popCount(promotions);

    if (chance == WHITE) {
        if ((pos->whiteCastle & CASTLE_FLAG_KING_SIDE) && !(F1G1 & allPieces) && !(F1G1 & threatened)) nMoves++;
        if ((pos->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) && !(B1D1 & allPieces) && !(C1D1 & threatened)) nMoves++;
    } else {
        if ((pos->blackCastle & CASTLE_FLAG_KING_SIDE) && !(F8G8 & allPieces) && !(F8G8 & threatened)) nMoves++;
        if ((pos->blackCastle & CASTLE_FLAG_QUEEN_SIDE) && !(B8D8 & allPieces) && !(C8D8 & threatened)) nMoves++;
    }
    
    uint64 kingMoves = kingAttacks(myKing);
    kingMoves &= ~(threatened | myPieces);
    nMoves += popCount(kingMoves);

    uint64 myKnights = (pos->knights & myPieces) & ~pinned;
    while (myKnights) {
        uint64 knight = getOne(myKnights);
        uint64 knightMoves = knightAttacks(knight) & ~myPieces;
        nMoves += popCount(knightMoves);
        myKnights ^= knight;
    }

    uint64 myBishops = pos->bishopQueens & myPieces;
    uint64 bishops = myBishops & pinned;
    while (bishops) {
        uint64 bishop = getOne(bishops);
        uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
        bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);
        nMoves += popCount(bishopMoves);
        bishops ^= bishop;
    }
    bishops = myBishops & ~pinned;
    while (bishops) {
        uint64 bishop = getOne(bishops);
        uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
        nMoves += popCount(bishopMoves);
        bishops ^= bishop;
    }

    uint64 myRooks = pos->rookQueens & myPieces;
    uint64 rooks = myRooks & pinned;
    while (rooks) {
        uint64 rook = getOne(rooks);
        uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
        rookMoves &= sqsInLine(bitScan(rook), kingIndex);
        nMoves += popCount(rookMoves);
        rooks ^= rook;
    }
    rooks = myRooks & ~pinned;
    while (rooks) {
        uint64 rook = getOne(rooks);
        uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
        nMoves += popCount(rookMoves);
        rooks ^= rook;
    }
    return nMoves;
}

#endif // MOVE_GENERATOR_H_
