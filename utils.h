#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include "chess.h" // This is essential as the function signatures use types defined here.

// This header contains the public declarations for the utility functions.
// The implementations are located in utils.cu.

class Utils {
public:
    // Reads a FEN string into the intermediate 0x88 board representation.
    static void readFENString(char fen[], BoardPosition *pos);

    // Converts the intermediate 0x88 board to the GPU-optimal HexaBitBoardPosition.
    static void board088ToHexBB(HexaBitBoardPosition *posBB, BoardPosition *pos088);

    // Converts the GPU-optimal HexaBitBoardPosition back to the intermediate 0x88 board.
    static void boardHexBBTo088(BoardPosition *pos088, HexaBitBoardPosition *posBB);

    // Displays the GPU board representation in a human-readable format.
    static void displayBoard(HexaBitBoardPosition *pos);
};

#endif // UTILS_H_INCLUDED
