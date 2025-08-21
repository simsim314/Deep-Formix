#ifndef MOVEGEN_DEVICE_CUH
#define MOVEGEN_DEVICE_CUH

#include "chess.h" // Needs FancyMagicEntry, ROOK_MAGIC_BITS, etc.

// ============================================================================
// EXTERNAL DECLARATIONS FOR GLOBAL GPU DEVICE-SIDE LOOKUP TABLES
//
// This header DECLARES global device-side variables using the 'extern'
// keyword. This tells any source file that includes it that these variables
// exist and what their types are, but that the actual memory for them is
// DEFINED in another source file (specifically, GlobalVars.cu).
// This is crucial to satisfy the C++/CUDA "One Definition Rule" and avoid
// linker errors.
// ============================================================================

// Zobrist keys for hashing
extern __device__ ZobristRandoms   gZob;
extern __device__ ZobristRandoms   gZob2;

// Bit mask tables
extern __device__ uint64 gBetween[64][64];
extern __device__ uint64 gLine[64][64];

// Pre-calculated attack tables for empty boards
extern __device__ uint64 gRookAttacks[64];
extern __device__ uint64 gBishopAttacks[64];
extern __device__ uint64 gQueenAttacks[64];
extern __device__ uint64 gKingAttacks[64];
extern __device__ uint64 gKnightAttacks[64];
extern __device__ uint64 gpawnAttacks[2][64];

// Magical Bitboard Tables
extern __device__ uint64 gRookAttacksMasked[64];
extern __device__ uint64 gBishopAttacksMasked[64];

// Plain magics
extern __device__ uint64 gRookMagics[64];
extern __device__ uint64 gBishopMagics[64];
extern __device__ uint64 gRookMagicAttackTables[64][1 << ROOK_MAGIC_BITS];
extern __device__ uint64 gBishopMagicAttackTables[64][1 << BISHOP_MAGIC_BITS];

// Fancy magics
extern __device__ uint64 g_fancy_magic_lookup_table[97264];
extern __device__ FancyMagicEntry g_bishop_magics_fancy[64];
extern __device__ FancyMagicEntry g_rook_magics_fancy[64];

// Byte lookup fancy magics
extern __device__ uint8  g_fancy_byte_magic_lookup_table[97264];
extern __device__ uint64 g_fancy_byte_BishopLookup[1428];
extern __device__ uint64 g_fancy_byte_RookLookup[4900];

#endif // MOVEGEN_DEVICE_CUH
