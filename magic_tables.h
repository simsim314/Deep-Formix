#ifndef MAGIC_TABLES_H_
#define MAGIC_TABLES_H_

#include "chess.h"
#include "cuda_adapters.cuh"

// CPU copy of all the below global variables are defined in GlobalVars.cpp
// bit mask containing squares between two given squares
extern uint64 Between[64][64];

// bit mask containing squares in the same 'line' as two given squares
extern uint64 Line[64][64];

// squares a piece can attack in an empty board
extern uint64 RookAttacks    [64];
extern uint64 BishopAttacks  [64];
extern uint64 QueenAttacks   [64];
extern uint64 KingAttacks    [64];
extern uint64 KnightAttacks  [64];
extern uint64 pawnAttacks[2] [64];

// magic lookup tables
// plain magics (Fancy magic lookup tables in FancyMagics.h)
extern uint64 rookMagics            [64];
extern uint64 bishopMagics          [64];

// same as RookAttacks and BishopAttacks, but corner bits masked off
extern uint64 RookAttacksMasked   [64];
extern uint64 BishopAttacksMasked [64];

extern uint64 rookMagicAttackTables      [64][1 << ROOK_MAGIC_BITS  ];    // 2 MB
extern uint64 bishopMagicAttackTables    [64][1 << BISHOP_MAGIC_BITS];    // 256 KB

// fancy and byte-lookup fancy magic tables
extern uint64 fancy_magic_lookup_table[97264];
extern FancyMagicEntry bishop_magics_fancy[64];
extern FancyMagicEntry rook_magics_fancy[64];
extern uint8  fancy_byte_magic_lookup_table[97264];  // 95 KB
extern uint64 fancy_byte_RookLookup        [4900] ;  // 39 K
extern uint64 fancy_byte_BishopLookup      [1428] ;  // 11 K


uint64 findRookMagicForSquare  (int square, uint64 magicAttackTable[], uint64 magic = 0, uint64 *uniqueAttackTable = NULL, uint8 *byteIndices = NULL, int *numUniqueAttacks = 0);
uint64 findBishopMagicForSquare(int square, uint64 magicAttackTable[], uint64 magic = 0, uint64 *uniqueAttackTable = NULL, uint8 *byteIndices = NULL, int *numUniqueAttacks = 0);

// set of random numbers for zobrist hashing
extern uint64           randoms[2000];    // set of 2000 random numbers (defined in randoms.cpp)
extern ZobristRandoms   zob;              // the random numbers actually used
extern ZobristRandoms   zob2;             // the random numbers actually used (second set - used only for 128 bit hashes)

#endif // MAGIC_TABLES_H_
