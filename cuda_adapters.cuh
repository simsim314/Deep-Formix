#ifndef CUDA_ADAPTERS_CUH_
#define CUDA_ADAPTERS_CUH_

#include "magic_tables.h"     // FOR HOST-SIDE LOOKUP TABLES
#include "movegen_device.cuh" // FOR DEVICE-SIDE LOOKUP TABLES

#ifdef __linux__
    #include <x86intrin.h>
    #define CPU_FORCE_INLINE inline
#else
    #include <intrin.h>
    #define CPU_FORCE_INLINE __forceinline
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#if USE_CONSTANT_MEMORY_FOR_LUT == 1 && defined(__CUDACC__)
#define CUDA_FAST_READ(x) (c ## x)
#else
#define CUDA_FAST_READ(x) (__ldg(&g ## x))
#endif

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqsInBetweenLUT(uint8 sq1, uint8 sq2)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBetween[sq1][sq2]);
#else
    return Between[sq1][sq2];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqsInLineLUT(uint8 sq1, uint8 sq2)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(Line[sq1][sq2]);
#else
    return Line[sq1][sq2];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqKnightAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(KnightAttacks[sq]);
#else
    return KnightAttacks[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqKingAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(KingAttacks[sq]);
#else
    return KingAttacks[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqRookAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(RookAttacks[sq]);
#else
    return RookAttacks[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqBishopAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(BishopAttacks[sq]);
#else
    return BishopAttacks[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqBishopAttacksMasked(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(BishopAttacksMasked[sq]);
#else
    return BishopAttacksMasked[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqRookAttacksMasked(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(RookAttacksMasked[sq]);
#else
    return RookAttacksMasked[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqRookMagics(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(RookMagics[sq]);
#else
    return rookMagics[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqBishopMagics(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return CUDA_FAST_READ(BishopMagics[sq]);
#else
    return bishopMagics[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqRookMagicAttackTables(uint8 sq, int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gRookMagicAttackTables[sq][index]);
#else
    return rookMagicAttackTables[sq][index];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqBishopMagicAttackTables(uint8 sq, int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBishopMagicAttackTables[sq][index]);
#else
    return bishopMagicAttackTables[sq][index];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sq_fancy_magic_lookup_table(int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&g_fancy_magic_lookup_table[index]);
#else
    return fancy_magic_lookup_table[index];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE FancyMagicEntry sq_bishop_magics_fancy(int sq)
{
#ifdef __CUDA_ARCH__
#if USE_CONSTANT_MEMORY_FOR_LUT == 1
    //return c_bishop_magics_fancy[sq];
    FancyMagicEntry op;
    // This is a way to read a struct from constant memory.
    // It assumes c_bishop_magics_fancy is defined in constant memory.
    // extern __constant__ FancyMagicEntry c_bishop_magics_fancy[64];
    // op.data = (((uint4 *)c_bishop_magics_fancy)[sq]);
    // Since c_... is not defined, we use __ldg for now.
    op.data = __ldg(&(((uint4 *)g_bishop_magics_fancy)[sq]));
    return op;
#else
    FancyMagicEntry op;
    op.data = __ldg(&(((uint4 *)g_bishop_magics_fancy)[sq]));
    return op;
#endif
#else
    return bishop_magics_fancy[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE FancyMagicEntry sq_rook_magics_fancy(int sq)
{
#ifdef __CUDA_ARCH__
#if USE_CONSTANT_MEMORY_FOR_LUT == 1
    //return c_rook_magics_fancy[sq];
    FancyMagicEntry op;
    // op.data = (((uint4 *)c_rook_magics_fancy)[sq]);
    op.data = __ldg(&(((uint4 *)g_rook_magics_fancy)[sq]));
    return op;
#else
    FancyMagicEntry op;
    op.data = __ldg(&(((uint4 *)g_rook_magics_fancy)[sq]));
    return op;
#endif
#else
    return rook_magics_fancy[sq];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint8 sq_fancy_byte_magic_lookup_table(int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&g_fancy_byte_magic_lookup_table[index]);
#else
    return fancy_byte_magic_lookup_table[index];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sq_fancy_byte_BishopLookup(int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&g_fancy_byte_BishopLookup[index]);
#else
    return fancy_byte_BishopLookup[index];
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sq_fancy_byte_RookLookup(int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&g_fancy_byte_RookLookup[index]);
#else
    return fancy_byte_RookLookup[index];
#endif
}

// --- ZOBRIST KEY MACROS (MOVED FROM MoveGeneratorBitboard.h) ---
#ifdef __CUDA_ARCH__
    #define ZOB_KEY1(x) (__ldg(&gZob.x))
    #define ZOB_KEY2(x) (__ldg(&gZob2.x))
#else
    #define ZOB_KEY1(x) (zob.x)
    #define ZOB_KEY2(x) (zob2.x)
#endif

// default is to use first set
#define ZOB_KEY ZOB_KEY1

#define ZOB_KEY_128(x) (HashKey128b(ZOB_KEY1(x), ZOB_KEY2(x)))


#endif // CUDA_ADAPTERS_CUH_
