#ifndef ATTACK_GENERATORS_H_
#define ATTACK_GENERATORS_H_

#include "bitboard_utils.h"
#include "magic_tables.h"

// ============================================================================
// KOGGE-STONE ATTACK GENERATORS
// These are low-level, branchless attack generators for sliding pieces.
// They are always defined here, as they are needed by Magics.cu for initialization.
// ============================================================================

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 northAttacks(uint64 gen, uint64 pro) {
    gen |= (gen << 8) & pro; pro &= (pro << 8);
    gen |= (gen << 16) & pro; pro &= (pro << 16);
    gen |= (gen << 32) & pro;
    return gen << 8;
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 southAttacks(uint64 gen, uint64 pro) {
    gen |= (gen >> 8) & pro; pro &= (pro >> 8);
    gen |= (gen >> 16) & pro; pro &= (pro >> 16);
    gen |= (gen >> 32) & pro;
    return gen >> 8;
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 eastAttacks(uint64 gen, uint64 pro) {
    pro &= ~FILEA;
    gen |= (gen << 1) & pro; pro &= (pro << 1);
    gen |= (gen << 2) & pro; pro &= (pro << 2);
    gen |= (gen << 4) & pro;
    return (gen << 1) & (~FILEA);
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 westAttacks(uint64 gen, uint64 pro) {
    pro &= ~FILEH;
    gen |= (gen >> 1) & pro; pro &= (pro >> 1);
    gen |= (gen >> 2) & pro; pro &= (pro >> 2);
    gen |= (gen >> 4) & pro;
    return (gen >> 1) & (~FILEH);
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 northEastAttacks(uint64 gen, uint64 pro) {
    pro &= ~FILEA;
    gen |= (gen << 9) & pro; pro &= (pro << 9);
    gen |= (gen << 18) & pro; pro &= (pro << 18);
    gen |= (gen << 36) & pro;
    return (gen << 9) & (~FILEA);
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 northWestAttacks(uint64 gen, uint64 pro) {
    pro &= ~FILEH;
    gen |= (gen << 7) & pro; pro &= (pro << 7);
    gen |= (gen << 14) & pro; pro &= (pro << 14);
    gen |= (gen << 28) & pro;
    return (gen << 7) & (~FILEH);
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 southEastAttacks(uint64 gen, uint64 pro) {
    pro &= ~FILEA;
    gen |= (gen >> 7) & pro; pro &= (pro >> 7);
    gen |= (gen >> 14) & pro; pro &= (pro >> 14);
    gen |= (gen >> 28) & pro;
    return (gen >> 7) & (~FILEA);
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 southWestAttacks(uint64 gen, uint64 pro) {
    pro &= ~FILEH;
    gen |= (gen >> 9) & pro; pro &= (pro >> 9);
    gen |= (gen >> 18) & pro; pro &= (pro >> 18);
    gen |= (gen >> 36) & pro;
    return (gen >> 9) & (~FILEH);
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 bishopAttacksKoggeStone(uint64 bishops, uint64 pro)
{
    return northEastAttacks(bishops, pro) |
           northWestAttacks(bishops, pro) |
           southEastAttacks(bishops, pro) |
           southWestAttacks(bishops, pro) ;
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 rookAttacksKoggeStone(uint64 rooks, uint64 pro)
{
    return northAttacks(rooks, pro) |
           southAttacks(rooks, pro) |
           eastAttacks (rooks, pro) |
           westAttacks (rooks, pro) ;
}

// ============================================================================
// HIGH-LEVEL ATTACK GENERATORS
// This section uses the compile-time switch to decide whether to use
// the fast magic bitboard lookups or the Kogge-Stone algorithm.
// ============================================================================

#if USE_SLIDING_LUT == 1
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 bishopAttacks(uint64 bishop, uint64 pro)
    {
        uint8 square = bitScan(bishop);
        uint64 occ = (~pro) & sqBishopAttacksMasked(square);

#if USE_FANCY_MAGICS == 1
#ifdef __CUDA_ARCH__
        FancyMagicEntry magicEntry = sq_bishop_magics_fancy(square);
        int index = (magicEntry.factor * occ) >> (64 - BISHOP_MAGIC_BITS);
#if USE_BYTE_LOOKUP_FANCY == 1
        int index2 = sq_fancy_byte_magic_lookup_table(magicEntry.position + index) + magicEntry.offset;
        return sq_fancy_byte_BishopLookup(index2);
#else
        return sq_fancy_magic_lookup_table(magicEntry.position + index);
#endif
#else // #ifdef __CUDA_ARCH__
        uint64 magic  = bishop_magics_fancy[square].factor;
        uint64 index = (magic * occ) >> (64 - BISHOP_MAGIC_BITS);
#if USE_BYTE_LOOKUP_FANCY == 1
        uint8 *table = &fancy_byte_magic_lookup_table[bishop_magics_fancy[square].position];
        int index2 = table[index] + bishop_magics_fancy[square].offset;
        return fancy_byte_BishopLookup[index2];
#else
        uint64 *table = &fancy_magic_lookup_table[bishop_magics_fancy[square].position];
        return table[index];
#endif
#endif // #ifdef __CUDA_ARCH__
#else // USE_FANCY_MAGICS == 1
        uint64 magic = sqBishopMagics(square);
        uint64 index = (magic * occ) >> (64 - BISHOP_MAGIC_BITS);
        return sqBishopMagicAttackTables(square, index);
#endif // USE_FANCY_MAGICS == 1
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 rookAttacks(uint64 rook, uint64 pro)
    {
        uint8 square = bitScan(rook);
        uint64 occ = (~pro) & sqRookAttacksMasked(square);

#if USE_FANCY_MAGICS == 1
#ifdef __CUDA_ARCH__
        FancyMagicEntry magicEntry = sq_rook_magics_fancy(square);
        int index = (magicEntry.factor * occ) >> (64 - ROOK_MAGIC_BITS);
#if USE_BYTE_LOOKUP_FANCY == 1
        int index2 = sq_fancy_byte_magic_lookup_table(magicEntry.position + index) + magicEntry.offset;
        return sq_fancy_byte_RookLookup(index2);
#else
        return sq_fancy_magic_lookup_table(magicEntry.position + index);
#endif
#else
        uint64 magic  = rook_magics_fancy[square].factor;
        uint64 index = (magic * occ) >> (64 - ROOK_MAGIC_BITS);
#if USE_BYTE_LOOKUP_FANCY == 1
        uint8 *table = &fancy_byte_magic_lookup_table[rook_magics_fancy[square].position];
        int index2 = table[index] + rook_magics_fancy[square].offset;
        return fancy_byte_RookLookup[index2];
#else
        uint64 *table = &fancy_magic_lookup_table[rook_magics_fancy[square].position];
        return table[index];
#endif
#endif
#else
        uint64 magic = sqRookMagics(square);
        uint64 index = (magic * occ) >> (64 - ROOK_MAGIC_BITS);
        return sqRookMagicAttackTables(square, index);
#endif
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 multiBishopAttacks(uint64 bishops, uint64 pro)
    {
        uint64 attacks = 0;
        while(bishops) {
            uint64 bishop = getOne(bishops);
            attacks |= bishopAttacks(bishop, pro);
            bishops ^= bishop;
        }
        return attacks;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 multiRookAttacks(uint64 rooks, uint64 pro)
    {
        uint64 attacks = 0;
        while(rooks) {
            uint64 rook = getOne(rooks);
            attacks |= rookAttacks(rook, pro);
            rooks ^= rook;
        }
        return attacks;
    }
#else
    // If not using lookup tables, just alias the generic names to the Kogge-Stone functions.
    #define bishopAttacks bishopAttacksKoggeStone
    #define rookAttacks   rookAttacksKoggeStone
    #define multiBishopAttacks bishopAttacksKoggeStone
    #define multiRookAttacks   rookAttacksKoggeStone
#endif

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 multiKnightAttacks(uint64 knights)
{
	uint64 attacks = 0;
	while(knights) {
		uint64 knight = getOne(knights);
		attacks |= sqKnightAttacks(bitScan(knight));
		knights ^= knight;
	}
	return attacks;
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 kingAttacks(uint64 kingSet)
{
    uint64 attacks = eastOne(kingSet) | westOne(kingSet);
    kingSet       |= attacks;
    attacks       |= northOne(kingSet) | southOne(kingSet);
    return attacks;
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 knightAttacks(uint64 knights) {
    uint64 l1 = (knights >> 1) & C64(0x7f7f7f7f7f7f7f7f);
    uint64 l2 = (knights >> 2) & C64(0x3f3f3f3f3f3f3f3f);
    uint64 r1 = (knights << 1) & C64(0xfefefefefefefefe);
    uint64 r2 = (knights << 2) & C64(0xfcfcfcfcfcfcfcfc);
    uint64 h1 = l1 | r1;
    uint64 h2 = l2 | r2;
    return (h1<<16) | (h1>>16) | (h2<<8) | (h2>>8);
}

#endif // ATTACK_GENERATORS_H_
