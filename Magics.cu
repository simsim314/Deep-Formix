// magic table initialization routines

// random generators and basic idea of finding magics taken from:
// http://chessprogramming.wikispaces.com/Looking+for+Magics 

#include "movegen_device.cuh"
#include "bitboard_utils.h"     // For getOne, popCount, BIT
#include "magic_tables.h"       // For RookAttacksMasked, BishopAttacksMasked
#include "attack_generators.h"  // For Kogge-Stone attack generators

uint64 random_uint64() 
{
      uint64 u1 = rand(), u2 = rand(), u3 = rand(), u4 = rand();
      return u1 | (u2 << 16) | (u3 << 32) | (u4 << 48);
}

uint64 random_uint64_sparse() 
{
    return random_uint64() & random_uint64() & random_uint64();
} 

// get i'th combo mask 
uint64 getOccCombo(uint64 mask, uint64 i)
{
    uint64 op = 0;
    while(i)
    {
        int bit = i % 2;
        uint64 opBit = getOne(mask);
        mask &= ~opBit;
        op |= opBit * bit;
        i = i >> 1;
    }

    return op;
}

uint64 findMagicCommon(uint64 occCombos[], uint64 attacks[], uint64 attackTable[], int numCombos, int bits, uint64 preCalculatedMagic = 0, uint64 *uniqueAttackTable = NULL, uint8 *byteIndices = NULL, int *numUniqueAttacks = NULL)
{
    uint64 magic = 0;
    while(1)
    {
        if (preCalculatedMagic)
        {
            magic = preCalculatedMagic;
        }
        else
        {
            for (int i=0; i < (1 << bits); i++)
            {
                attackTable[i] = 0; // unused entry
            }

            magic = random_uint64_sparse();
        }

        // try all possible occupancy combos and check for collisions
        int i = 0;
        for (i = 0; i < numCombos; i++)
        {
            uint64 index = (magic * occCombos[i]) >> (64 - bits);
            if (attackTable[index] == 0)
            {
                uint64 attackSet = attacks[i];
                attackTable[index] = attackSet;

                // fill in the byte lookup table also
                if (numUniqueAttacks)
                {
                    // search if this attack set is already present in uniqueAttackTable
                    int j = 0;
                    for (j = 0; j < *numUniqueAttacks; j++)
                    {
                        if (uniqueAttackTable[j] == attackSet)
                        {
                            byteIndices[index] = j;
                            break;
                        }
                    }

                    // add new unique attack entry if not found
                    if (j == *numUniqueAttacks)
                    {
                        uniqueAttackTable[*numUniqueAttacks] = attackSet;
                        byteIndices[index] = *numUniqueAttacks;
                        (*numUniqueAttacks)++;
                    }
                }
            }
            else
            {
                // mismatching entry already found
                if (attackTable[index] != attacks[i])
                    break;
            }
        }

        if (i == numCombos)
            break;
        else
            assert(preCalculatedMagic == 0);
    }
    return magic;
}

uint64 findRookMagicForSquare(int square, uint64 magicAttackTable[], uint64 preCalculatedMagic, uint64 *uniqueAttackTable, uint8 *byteIndices, int *numUniqueAttacks)
{
    uint64 mask = RookAttacksMasked[square];
    uint64 thisSquare = BIT(square);

    int numBits   =  popCount(mask);
    int numCombos = (1 << numBits);
    
    uint64 occCombos[4096];
    uint64 attacks[4096];

    for (int i=0; i < numCombos; i++)
    {
        occCombos[i] = getOccCombo(mask, i);
        attacks[i]   = rookAttacksKoggeStone(thisSquare, ~occCombos[i]);
    }

    return findMagicCommon(occCombos, attacks, magicAttackTable, numCombos, ROOK_MAGIC_BITS, preCalculatedMagic, uniqueAttackTable, byteIndices, numUniqueAttacks);
}

uint64 findBishopMagicForSquare(int square, uint64 magicAttackTable[], uint64 preCalculatedMagic, uint64 *uniqueAttackTable, uint8 *byteIndices, int *numUniqueAttacks)
{
    uint64 mask = BishopAttacksMasked[square];
    uint64 thisSquare = BIT(square);

    int numBits   =  popCount(mask);
    int numCombos = (1 << numBits);
    
    uint64 occCombos[4096];
    uint64 attacks[4096];

    for (int i=0; i < numCombos; i++)
    {
        occCombos[i] = getOccCombo(mask, i);
        attacks[i]   = bishopAttacksKoggeStone(thisSquare, ~occCombos[i]);
    }

    return findMagicCommon(occCombos, attacks, magicAttackTable, numCombos, BISHOP_MAGIC_BITS, preCalculatedMagic, uniqueAttackTable, byteIndices, numUniqueAttacks);
}

// only for testing
#if 0
uint64 rookMagicAttackTables[64][1 << ROOK_MAGIC_BITS];
uint64 bishopMagicAttackTables[64][1 << BISHOP_MAGIC_BITS];

void findBishopMagics()
{
    printf("\n\nBishop Magics: ...");
    for (int square = A1; square <= H8; square++)
    {
        uint64 magic = findBishopMagicForSquare(square, bishopMagicAttackTables[square]);
        printf("\nSquare: %c%c, Magic: %X%X", 'A' + (square%8), '1' + (square / 8), HI(magic), LO(magic));
    }
}

void findRookMagics()
{
    printf("\n\nRook Magics: ...");
    for (int square = A1; square <= H8; square++)
    {
        uint64 magic = findRookMagicForSquare(square, rookMagicAttackTables[square]);
        printf("\nSquare: %c%c, Magic: %X%X", 'A' + (square%8), '1' + (square / 8), HI(magic), LO(magic));
    }
}

void findMagics()
{
    srand (time(NULL));
    findBishopMagics();
    findRookMagics();
}
#endif
