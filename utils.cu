#include <cstring>
#include <cctype>
#include "utils.h"

#include <cstdio>
#include <cinttypes>

#include "bitboard_constants.h"  // for RANK1, RANK8, etc.

// Converts the GPU-optimal HexaBitBoardPosition back to the intermediate 0x88 board.
void Utils::boardHexBBTo088(BoardPosition *pos088, HexaBitBoardPosition *posBB) {
    memset(pos088, 0, sizeof(BoardPosition));

    // Mask away the low byte of the pawn bitboard (flags overlap a1/b1 etc.)
    // Only keep *real* pawn squares (ranks 2..7)
    uint64 pawnSquares = posBB->pawns & RANKS2TO7;

    // Derive queens from bishopQueens & rookQueens
    uint64 queens = posBB->bishopQueens & posBB->rookQueens;

    // Combine all piece planes into one "occupied" set
    uint64 allPieces = posBB->kings
                     | pawnSquares
                     | posBB->knights
                     | posBB->bishopQueens
                     | posBB->rookQueens;

    for(uint8 i = 0; i < 64; ++i) {
        if (BIT(i) & allPieces) {
            uint8 color = (BIT(i) & posBB->whitePieces) ? WHITE : BLACK;
            uint8 piece = 0;

            if      (BIT(i) & posBB->kings)        piece = KING;
            else if (BIT(i) & posBB->knights)      piece = KNIGHT;
            else if (BIT(i) & pawnSquares)         piece = PAWN;       // now safe
            else if (BIT(i) & queens)              piece = QUEEN;
            else if (BIT(i) & posBB->rookQueens)   piece = ROOK;
            else if (BIT(i) & posBB->bishopQueens) piece = BISHOP;

            pos088->board[INDEX088(i >> 3, i & 7)] = COLOR_PIECE(color, piece);
        }
    }

    // Copy flags directly
    pos088->chance          = posBB->chance;
    pos088->blackCastle     = posBB->blackCastle;
    pos088->whiteCastle     = posBB->whiteCastle;
    pos088->enPassent       = posBB->enPassent;
    pos088->halfMoveCounter = posBB->halfMoveCounter;
}

// ─── helpers ────────────────────────────────────────────────────────────────
static inline void printBitboardGrid(const char* name, uint64 bb)
{
    printf("%s\n", name);
    printf("  +---+---+---+---+---+---+---+---+\n");
    for (int r = 7; r >= 0; --r) {
        printf("%d |", r + 1);
        for (int f = 0; f < 8; ++f) {
            uint8 bit = (bb >> (r * 8 + f)) & 1ULL;
            printf(" %d |", bit);
        }
        printf("\n  +---+---+---+---+---+---+---+---+\n");
    }
    printf("    a   b   c   d   e   f   g   h\n\n");
}

static inline void printEightBits(uint8 v)
{
    for (int b = 7; b >= 0; --b) putchar(((v >> b) & 1U) ? '1' : '0');
}

// ─── main routine ───────────────────────────────────────────────────────────
void fullPrint(const HexaBitBoardPosition* posBB)
{
    puts("\n======================= FULL POSITION DUMP =======================");

    /* 1. Pretty board (piece letters) */
    Utils::displayBoard(const_cast<HexaBitBoardPosition*>(posBB));

    /* 2. Raw 8‑bit value of every square (like your debug print) */
    BoardPosition tmp088;
    Utils::boardHexBBTo088(&tmp088, const_cast<HexaBitBoardPosition*>(posBB));

    puts("\nRaw 0x88 contents (binary per square):");
    printf("  +---+---+---+---+---+---+---+---+\n");
    for (int r = 7; r >= 0; --r) {
        printf("%d |", r + 1);
        for (int f = 0; f < 8; ++f) {
            uint8 v = tmp088.board[INDEX088(r, f)];
            printEightBits(v);
            printf("|");
        }
        printf("\n  +---+---+---+---+---+---+---+---+\n");
    }
    printf("    a   b   c   d   e   f   g   h\n\n");

    /* 3. Per‑plane bitboards ------------------------------------------------*/
    uint64 queens = posBB->bishopQueens & posBB->rookQueens;

    printBitboardGrid("whitePieces (set = White square)", posBB->whitePieces);
    printBitboardGrid("pawns (low byte = flags!)",         posBB->pawns);
    printBitboardGrid("knights",                           posBB->knights);
    printBitboardGrid("bishopQueens",                      posBB->bishopQueens);
    printBitboardGrid("rookQueens",                        posBB->rookQueens);
    printBitboardGrid("queens  (derived)",                 queens);
    printBitboardGrid("kings",                             posBB->kings);

    /* 5. Flags / game‑state -------------------------------------------------*/
    printf("FLAGS / GAME‑STATE:\n");
    printf("  side to move (chance) : %u (%s)\n",
           posBB->chance, posBB->chance ? "Black" : "White");

    printf("  castling rights       : white 0x%x  black 0x%x   "
           "(K‑side = 1, Q‑side = 2)\n",
           posBB->whiteCastle, posBB->blackCastle);

    char epFileChar = posBB->enPassent ? ('a' + posBB->enPassent - 1) : '-';

    printf("  en‑passant file (+1)  : %u  (%c)\n",
           posBB->enPassent, epFileChar);

    printf("  half‑move counter     : %u\n\n", posBB->halfMoveCounter);

    puts("===================== END OF POSITION DUMP ======================\n");
}

// This helper is only used within this file, so it's declared static.
static uint8 getPieceCode(char piece) {
    if (islower(piece)) {
        const char* pieces = "pnbrqk";
        const char* p = strchr(pieces, piece);
        if (p) return COLOR_PIECE(BLACK, (p - pieces) + 1);
    } else {
        const char* pieces = "PNBRQK";
        const char* p = strchr(pieces, piece);
        if (p) return COLOR_PIECE(WHITE, (p - pieces) + 1);
    }
    return EMPTY_SQUARE;
}

// Another internal helper.
static char getPieceChar(uint8 code) {
    const char pieceCharMapping[] = {'.', 'P', 'N', 'B', 'R', 'Q', 'K'};
    if (code == EMPTY_SQUARE) return '.';
    uint8 piece = PIECE(code);
    char pieceChar = pieceCharMapping[piece];
    if (COLOR(code) == BLACK) {
        pieceChar = tolower(pieceChar);
    }
    return pieceChar;
}

// Converts the intermediate 0x88 board to the GPU-optimal HexaBitBoardPosition.
void Utils::board088ToHexBB(HexaBitBoardPosition *posBB, BoardPosition *pos088) {
    memset(posBB, 0, sizeof(HexaBitBoardPosition));
    uint64 queens = 0;
    for (uint8 i = 0; i < 64; i++) {

        uint8 rank = i >> 3;
        uint8 file = i & 7;
        uint8 index088 = INDEX088(rank, file);
        uint8 colorpiece = pos088->board[index088];
        if (colorpiece != EMPTY_SQUARE) {
            uint8 color = COLOR(colorpiece);
            uint8 piece = PIECE(colorpiece);
            if (color == WHITE) posBB->whitePieces |= BIT(i);
            switch (piece) {
                case PAWN:   posBB->pawns |= BIT(i); break;
                case KNIGHT: posBB->knights |= BIT(i); break;
                case BISHOP: posBB->bishopQueens |= BIT(i); break;
                case ROOK:   posBB->rookQueens |= BIT(i); break;
                case QUEEN:  queens |= BIT(i); break;
                case KING:   posBB->kings |= BIT(i); break;
            }
        }
    }


    posBB->bishopQueens |= queens;
    posBB->rookQueens |= queens;
    posBB->chance = pos088->chance;
    posBB->blackCastle = pos088->blackCastle;
    posBB->whiteCastle = pos088->whiteCastle;
    posBB->enPassent = pos088->enPassent;
    posBB->halfMoveCounter = pos088->halfMoveCounter;
    //fullPrint(posBB);

}


// Read a FEN string into the intermediate 0x88 board representation.
void Utils::readFENString(char fen[], BoardPosition *pos) {
    memset(pos, 0, sizeof(BoardPosition));
    int i = 0;
    for (int r = 7; r >= 0; --r) {
        int f = 0;
        while (f < 8 && fen[i] != '/' && fen[i] != ' ' && fen[i] != '\0') {
            char c = fen[i++];
            if (c >= '1' && c <= '8') {
                f += c - '0';
            } else {
                pos->board[INDEX088(r, f)] = getPieceCode(c);
                f++;
            }
        }
        if (fen[i] == '/') i++;
    }
    while(fen[i] == ' ') i++;
    pos->chance = (fen[i] == 'b') ? BLACK : WHITE;
    i++;
    while(fen[i] == ' ') i++;
    pos->whiteCastle = pos->blackCastle = 0;
    while (fen[i] != ' ') {
        switch (fen[i++]) {
            case 'K': pos->whiteCastle |= CASTLE_FLAG_KING_SIDE; break;
            case 'Q': pos->whiteCastle |= CASTLE_FLAG_QUEEN_SIDE; break;
            case 'k': pos->blackCastle |= CASTLE_FLAG_KING_SIDE; break;
            case 'q': pos->blackCastle |= CASTLE_FLAG_QUEEN_SIDE; break;
        }
    }
    while(fen[i] == ' ') i++;
    pos->enPassent = (fen[i] >= 'a' && fen[i] <= 'h') ? (fen[i] - 'a' + 1) : 0;
}

// Display the GPU board representation in a human-readable format.
void Utils::displayBoard(HexaBitBoardPosition *posBB) {
    BoardPosition pos088;
    boardHexBBTo088(&pos088, posBB);

    printf("\n  +---+---+---+---+---+---+---+---+\n");
    for (int r = 7; r >= 0; --r) {
        printf("%d |", r + 1);
        for (int f = 0; f < 8; ++f) {
            char piece = getPieceChar(pos088.board[INDEX088(r, f)]);
            printf(" %c |", piece);
        }
        printf("\n  +---+---+---+---+---+---+---+---+\n");
    }
    printf("    a   b   c   d   e   f   g   h\n");
}
