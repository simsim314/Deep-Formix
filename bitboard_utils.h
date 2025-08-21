#ifndef BITBOARD_UTILS_H_
#define BITBOARD_UTILS_H_

#include "bitboard_constants.h"
#include "cuda_adapters.cuh"

// ──────────────────────────────────────────────────────────────────────────────
//  Geometry helpers – pure functions, usable from host & device
// ──────────────────────────────────────────────────────────────────────────────

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE
uint64 squaresInLine(uint8 sq1, uint8 sq2)
{
    int fd =   (sq2 & 7) - (sq1 & 7);
    int rd =  ((sq2 | 7) -  sq1) >> 3;

    uint8 f = sq1 & 7;
    uint8 r = sq1 >> 3;

    if (fd == 0)                 return FILEA  << f;            // same file
    if (rd == 0)                 return RANK1  << (r * 8);      // same rank
    if (fd - rd == 0)            // main diagonal
        return (r >= f ? DIAGONAL_A1H8 << ((r - f) * 8)
                       : DIAGONAL_A1H8 >> ((f - r) * 8));
    if (fd + rd == 0) {          // anti‑diagonal
        int sh = (r + f - 7) * 8;
        return sh >= 0 ? DIAGONAL_A8H1 << sh : DIAGONAL_A8H1 >> -sh;
    }
    return 0;                    // not colinear
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint8 popCount(uint64 x)
{
#ifdef __CUDA_ARCH__
    return __popcll(x);
#elif USE_POPCNT == 1
#ifdef _WIN64
    return _mm_popcnt_u64(x);
#elif __linux__
    return _mm_popcnt_u64(x);
#else
    uint32 lo = (uint32)  x;
    uint32 hi = (uint32) (x >> 32);
    return _mm_popcnt_u32(lo) + _mm_popcnt_u32(hi);
#endif
#else
    const uint64 k1 = C64(0x5555555555555555);
    const uint64 k2 = C64(0x3333333333333333);
    const uint64 k4 = C64(0x0f0f0f0f0f0f0f0f);
    const uint64 kf = C64(0x0101010101010101);

    // taken from chess prgramming wiki: http://chessprogramming.wikispaces.com/Population+Count

    x =  x       - ((x >> 1)  & k1); /* put count of each 2 bits into those 2 bits */
    x = (x & k2) + ((x >> 2)  & k2); /* put count of each 4 bits into those 4 bits */
    x = (x       +  (x >> 4)) & k4 ; /* put count of each 8 bits into those 8 bits */
    x = (x * kf) >> 56;              /* returns 8 most significant bits of x + (x<<8) + (x<<16) + (x<<24) + ...  */

    return (uint8) x;
#endif
}
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint8 bitScan(uint64 x)
{
#ifdef __CUDA_ARCH__
    // __ffsll(x) returns position from 1 to 64 instead of 0 to 63
    return __ffsll(x) - 1;
#elif _WIN64
   unsigned long index;
   assert (x != 0);
   _BitScanForward64(&index, x);
   return (uint8) index;
#elif __linux__
    return __builtin_ffsll(x) - 1;
#else
#if USE_HW_BITSCAN == 1
    uint32 lo = (uint32)  x;
    uint32 hi = (uint32) (x >> 32);
    unsigned long id;

    if (lo)
        _BitScanForward(&id, lo);
    else
    {
        _BitScanForward(&id, hi);
        id += 32;
    }

    return (uint8) id;
#else
    const int index64[64] = {
        0,  1, 48,  2, 57, 49, 28,  3,
       61, 58, 50, 42, 38, 29, 17,  4,
       62, 55, 59, 36, 53, 51, 43, 22,
       45, 39, 33, 30, 24, 18, 12,  5,
       63, 47, 56, 27, 60, 41, 37, 16,
       54, 35, 52, 21, 44, 32, 23, 11,
       46, 26, 40, 15, 34, 20, 31, 10,
       25, 14, 19,  9, 13,  8,  7,  6
    };
    const uint64 debruijn64 = C64(0x03f79d71b4cb0a89);
    assert (x != 0);
    return index64[((x & -x) * debruijn64) >> 58];
#endif
#endif
}
    // move the bits in the bitboard one square in the required direction

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northOne(uint64 x)
    {
        return x << 8;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southOne(uint64 x)
    {
        return x >> 8;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 eastOne(uint64 x)
    {
        return (x << 1) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 westOne(uint64 x)
    {
        return (x >> 1) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northEastOne(uint64 x)
    {
        return (x << 9) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northWestOne(uint64 x)
    {
        return (x << 7) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southEastOne(uint64 x)
    {
        return (x >> 7) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southWestOne(uint64 x)
    {
        return (x >> 9) & (~FILEH);
    }

    // gets one bit (the LSB) from a bitboard
    // returns a bitboard containing that bit
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 getOne(uint64 x)
    {
        return x & (-x);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static bool isMultiple(uint64 x)
    {
        return x ^ getOne(x);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static bool isSingular(uint64 x)
    {
        return !isMultiple(x);
    }
    // finds the squares in between the two given squares
    // taken from
    // http://chessprogramming.wikispaces.com/Square+Attacked+By#Legality Test-In Between-Pure Calculation
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 squaresInBetween(uint8 sq1, uint8 sq2)
    {
        const uint64 m1 = C64(0xFFFFFFFFFFFFFFFF);
        const uint64 a2a7 = C64(0x0001010101010100);
        const uint64 b2g7 = C64(0x0040201008040200);
        const uint64 h1b7 = C64(0x0002040810204080);
        uint64 btwn, line, rank, file;

        btwn  = (m1 << sq1) ^ (m1 << sq2);
        file  =   (sq2 & 7) - (sq1   & 7);
        rank  =  ((sq2 | 7) -  sq1) >> 3 ;
        line  =      (   (file  &  7) - 1) & a2a7; // a2a7 if same file
        line += 2 * ((   (rank  &  7) - 1) >> 58); // b1g1 if same rank
        line += (((rank - file) & 15) - 1) & b2g7; // b2g7 if same diagonal
        line += (((rank + file) & 15) - 1) & h1b7; // h1b7 if same antidiag
        line *= btwn & -btwn; // mul acts like shift by smaller square
        return line & btwn;   // return the bits on that line inbetween
    }


    // name‑stable wrappers
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE
    uint64 sqsInBetween(uint8 a,uint8 b){ return squaresInBetween(a,b); }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE
    uint64 sqsInLine(uint8 a,uint8 b)   { return squaresInLine(a,b); }

#endif // BITBOARD_UTILS_H_
