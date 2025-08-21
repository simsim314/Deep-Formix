// src/uci.cu
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <climits>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "uci.cuh"
#include "search.cuh"
#include "chess.h"
#include "MoveGeneratorBitboard.h"
#include "utils.h"
#include "switches.h" // Needed for the default network path macros

namespace UCI {

using std::string;
using std::vector;
using std::ostringstream;

// Define and initialize the global path variables with the compiled-in defaults.
std::string g_nnue_model_path = NNUE_MODEL_PATH;
std::string g_nnue_mapping_path = NNUE_MAPPING_PATH;

// Global engine board (project-provided)
HexaBitBoardPosition g_board;

// Chronological repetition keys; last element = current position
static std::vector<std::string> position_history;

// ---------- Helpers: build FEN from HexaBitBoardPosition ----------
static inline char piece_char_from_mask(bool is_white, int piece_type) {
    char c = '?';
    switch (piece_type) {
        case PAWN:   c = 'p'; break;
        case KNIGHT: c = 'n'; break;
        case BISHOP: c = 'b'; break;
        case ROOK:   c = 'r'; break;
        case QUEEN:  c = 'q'; break;
        case KING:   c = 'k'; break;
        default:     c = '?'; break;
    }
    if (is_white) c = static_cast<char>(std::toupper((unsigned char)c));
    return c;
}

// Convert a HexaBitBoardPosition into a FEN string.
// Assumes bit 0 == a1, ... bit 63 == h8.
static std::string get_fen_from_hexa(const HexaBitBoardPosition& board) {
    char board_chars[8][8];
    for (int r = 0; r < 8; ++r)
        for (int f = 0; f < 8; ++f)
            board_chars[r][f] = ' ';

    auto bit_test = [](uint64 bb, int idx)->bool { return ((bb >> idx) & 1ULL) != 0ULL; };

    for (int idx = 0; idx < 64; ++idx) {
        bool is_white = bit_test(board.whitePieces, idx);
        bool is_king  = bit_test(board.kings, idx);
        bool is_knight= bit_test(board.knights, idx);
        bool is_pawn  = bit_test(board.pawns, idx);
        bool in_rookq = bit_test(board.rookQueens, idx);
        bool in_bishq = bit_test(board.bishopQueens, idx);

        int piece = EMPTY_SQUARE;
        if (is_king) {
            piece = KING;
        } else if (is_pawn) {
            piece = PAWN;
        } else if (is_knight) {
            piece = KNIGHT;
        } else {
            if (in_rookq && in_bishq) piece = QUEEN;
            else if (in_rookq) piece = ROOK;
            else if (in_bishq) piece = BISHOP;
            else piece = EMPTY_SQUARE;
        }

        if (piece != EMPTY_SQUARE) {
            int rank_index = idx / 8; // 0 == rank1
            int file_index = idx % 8;
            int row = 7 - rank_index; // row 0 => rank8
            int col = file_index;
            board_chars[row][col] = piece_char_from_mask(is_white, piece);
        }
    }

    std::ostringstream fen;
    for (int r = 0; r < 8; ++r) {
        int empty_run = 0;
        for (int f = 0; f < 8; ++f) {
            char c = board_chars[r][f];
            if (c == ' ' || c == '?') {
                ++empty_run;
            } else {
                if (empty_run) { fen << empty_run; empty_run = 0; }
                fen << c;
            }
        }
        if (empty_run) fen << empty_run;
        if (r != 7) fen << '/';
    }

    char side = (board.chance == WHITE) ? 'w' : 'b';
    fen << ' ' << side << ' ';

    std::string castle;
    if ((board.whiteCastle & CASTLE_FLAG_KING_SIDE) != 0)  castle.push_back('K');
    if ((board.whiteCastle & CASTLE_FLAG_QUEEN_SIDE) != 0) castle.push_back('Q');
    if ((board.blackCastle & CASTLE_FLAG_KING_SIDE) != 0)  castle.push_back('k');
    if ((board.blackCastle & CASTLE_FLAG_QUEEN_SIDE) != 0) castle.push_back('q');
    if (castle.empty()) castle = "-";
    fen << castle << ' ';

    if (board.enPassent != 0 && (board.enPassent <= 8)) {
        int file = (int)board.enPassent - 1; // 0..7
        char rank_ch = (board.chance == BLACK) ? '3' : '6';
        char file_ch = 'a' + file;
        fen << file_ch << rank_ch << ' ';
    } else {
        fen << "- ";
    }

    int halfmove = static_cast<int>(board.halfMoveCounter);
    fen << halfmove << ' ';
    fen << 1; // fullmove number (not stored in Hexa; use 1)
    return fen.str();
}

// Convenience: current global g_board to FEN
static std::string get_fen_from_gboard() {
    return get_fen_from_hexa(g_board);
}

// Extract repetition relevant fields from FEN.
// For threefold repetition, only these matter:
// 1. Piece positions (field 1)
// 2. Side to move (field 2)
// Castling rights and en passant do NOT affect repetition detection
static inline std::string repetition_key_from_fen(const std::string& fen) {
    std::istringstream iss(fen);
    std::string piece_positions, side_to_move;
    
    // Read only the 2 fields we need for repetition detection
    if (!(iss >> piece_positions)) piece_positions = "";
    if (!(iss >> side_to_move)) side_to_move = "w"; // default to white
    
    // Build the repetition key from only these 2 fields
    return piece_positions + " " + side_to_move;
}

// History helpers
static void push_current_position_to_history() {
    std::string fen = get_fen_from_gboard();
    position_history.push_back(repetition_key_from_fen(fen));
}
static int count_key_occurrences(const std::string& key) {
    return static_cast<int>(std::count(position_history.begin(), position_history.end(), key));
}

// ---------- Rebuild history from start FEN + UCI moves and return final position ----------
// This simulates moves on a local HexaBitBoardPosition 'tmp', pushes the repetition key
// for the start position and after each move into position_history, and returns tmp
// (the final position). It does NOT modify global g_board.
static HexaBitBoardPosition rebuild_history_from_position(const std::string& fen_start, const std::vector<std::string>& uci_moves) {
    position_history.clear();
    position_history.push_back(repetition_key_from_fen(fen_start));

    // Build initial tmp from fen_start
    BoardPosition pos088;
    Utils::readFENString(const_cast<char*>(fen_start.c_str()), &pos088);
    HexaBitBoardPosition tmp;
    Utils::board088ToHexBB(&tmp, &pos088);

    // For each UCI move token, find a matching legal move on 'tmp' and apply it.
    for (const auto& mv : uci_moves) {
        CMove legal_moves[MAX_MOVES];
        uint32 num_moves;
        if (tmp.chance == WHITE) num_moves = MoveGeneratorBitboard::generateMoves<WHITE>(&tmp, legal_moves);
        else num_moves = MoveGeneratorBitboard::generateMoves<BLACK>(&tmp, legal_moves);

        bool applied_local = false;
        // Try to match by converting each legal move to string (SEARCH::moveToString)
        for (uint32 i = 0; i < num_moves; ++i) {
            char buf[16] = {0};
            SEARCH::moveToString(legal_moves[i], buf);
            if (mv == std::string(buf)) {
                uint64 unused_hash = 0ULL;
                if (tmp.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&tmp, unused_hash, legal_moves[i]);
                else MoveGeneratorBitboard::makeMove<BLACK, false>(&tmp, unused_hash, legal_moves[i]);
                applied_local = true;
                break;
            }
        }

        // If not matched by string, fall back to parsing UCI token (from/to/promotion)
        if (!applied_local) {
            if (mv.length() >= 4) {
                uint8 from_sq = (mv[0] - 'a') + (mv[1] - '1') * 8;
                uint8 to_sq   = (mv[2] - 'a') + (mv[3] - '1') * 8;
                char promo_ch = (mv.length() >= 5) ? mv[4] : 0;
                for (uint32 i = 0; i < num_moves; ++i) {
                    uint8 f = legal_moves[i].getFrom();
                    uint8 t = legal_moves[i].getTo();
                    if (f == from_sq && t == to_sq) {
                        if (promo_ch) {
                            char promotion_char = static_cast<char>(std::tolower((unsigned char)promo_ch));
                            uint8 flags = legal_moves[i].getFlags();
                            if ((promotion_char == 'q' && (flags == CM_FLAG_QUEEN_PROMOTION || flags == CM_FLAG_QUEEN_PROMO_CAP)) ||
                                (promotion_char == 'r' && (flags == CM_FLAG_ROOK_PROMOTION  || flags == CM_FLAG_ROOK_PROMO_CAP)) ||
                                (promotion_char == 'b' && (flags == CM_FLAG_BISHOP_PROMOTION|| flags == CM_FLAG_BISHOP_PROMO_CAP)) ||
                                (promotion_char == 'n' && (flags == CM_FLAG_KNIGHT_PROMOTION|| flags == CM_FLAG_KNIGHT_PROMO_CAP))) {
                                uint64 unused_hash = 0ULL;
                                if (tmp.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&tmp, unused_hash, legal_moves[i]);
                                else MoveGeneratorBitboard::makeMove<BLACK, false>(&tmp, unused_hash, legal_moves[i]);
                                applied_local = true;
                                break;
                            }
                        } else {
                            uint64 unused_hash = 0ULL;
                            if (tmp.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&tmp, unused_hash, legal_moves[i]);
                            else MoveGeneratorBitboard::makeMove<BLACK, false>(&tmp, unused_hash, legal_moves[i]);
                            applied_local = true;
                            break;
                        }
                    }
                }
            }
        }

        // Push repetition key for the new tmp (after the move), even if we failed to apply a move
        position_history.push_back(repetition_key_from_fen(get_fen_from_hexa(tmp)));
    }

    return tmp;
}

// ---------- Board setters / moves ----------
void convert_and_set_board(const char* fen) {
    BoardPosition pos088;
    Utils::readFENString(const_cast<char*>(fen), &pos088);
    Utils::board088ToHexBB(&g_board, &pos088);
}

void newgame() {
    convert_and_set_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    position_history.clear();
    push_current_position_to_history();
}

bool apply_move_internal(const std::string& moveToken) {
    if (moveToken.length() < 4) return false;
    uint8 from_sq = (moveToken[0] - 'a') + (moveToken[1] - '1') * 8;
    uint8 to_sq = (moveToken[2] - 'a') + (moveToken[3] - '1') * 8;
    CMove legal_moves[MAX_MOVES];
    uint32 num_moves;
    uint8 chance = g_board.chance;

    if (chance == WHITE) num_moves = MoveGeneratorBitboard::generateMoves<WHITE>(&g_board, legal_moves);
    else num_moves = MoveGeneratorBitboard::generateMoves<BLACK>(&g_board, legal_moves);

    CMove user_move;
    bool found_move = false;
    for (uint32 i = 0; i < num_moves; ++i) {
        if (legal_moves[i].getFrom() == from_sq && legal_moves[i].getTo() == to_sq) {
            if (moveToken.length() == 5) {
                char promotion_char = static_cast<char>(std::tolower((unsigned char)moveToken[4]));
                uint8 flags = legal_moves[i].getFlags();
                if ((promotion_char == 'q' && (flags == CM_FLAG_QUEEN_PROMOTION || flags == CM_FLAG_QUEEN_PROMO_CAP)) ||
                   (promotion_char == 'r' && (flags == CM_FLAG_ROOK_PROMOTION || flags == CM_FLAG_ROOK_PROMO_CAP)) ||
                   (promotion_char == 'b' && (flags == CM_FLAG_BISHOP_PROMOTION || flags == CM_FLAG_BISHOP_PROMO_CAP)) ||
                   (promotion_char == 'n' && (flags == CM_FLAG_KNIGHT_PROMOTION || flags == CM_FLAG_KNIGHT_PROMO_CAP))) {
                    user_move = legal_moves[i];
                    found_move = true;
                    break;
                }
            } else {
                user_move = legal_moves[i];
                found_move = true;
                break;
            }
        }
    }

    if (found_move) {
        uint64 unused_hash = 0ULL;
        if (chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&g_board, unused_hash, user_move);
        else MoveGeneratorBitboard::makeMove<BLACK, false>(&g_board, unused_hash, user_move);

        push_current_position_to_history();
        return true;
    }
    return false;
}

// ---------- position handler ----------
void handle_position(std::istringstream &is) {
    std::string token, fen;
    is >> token;
    std::vector<std::string> move_list_uci;
    if (token == "startpos") {
        // collect optional "moves" token and moves
        std::string rest;
        while (is >> rest) {
            if (rest == "moves") {
                std::string mv;
                while (is >> mv) move_list_uci.push_back(mv);
                break;
            }
        }
        // rebuild history and get final position (no duplicate pushes)
        HexaBitBoardPosition final_pos = rebuild_history_from_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move_list_uci);
        // set global board to final position (history already contains start + moves)
        g_board = final_pos;
        return;
    } else if (token == "fen") {
        fen.clear();
        std::string part;
        std::vector<std::string> fen_parts;
        while (is >> part && part != "moves") {
            fen_parts.push_back(part);
        }
        {
            std::ostringstream of;
            for (size_t i = 0; i < fen_parts.size(); ++i) {
                if (i) of << " ";
                of << fen_parts[i];
            }
            fen = of.str();
        }
        // collect remaining moves
        std::string mv;
        while (is >> mv) move_list_uci.push_back(mv);
        // rebuild history and obtain final position
        HexaBitBoardPosition final_pos = rebuild_history_from_position(fen, move_list_uci);
        g_board = final_pos;
        return;
    } else {
        // fallback: treat rest of line as fen
        std::ostringstream of;
        of << token;
        std::string rest;
        while (is >> rest) {
            of << " " << rest;
        }
        fen = of.str();
        convert_and_set_board(fen.c_str());
        position_history.clear();
        push_current_position_to_history();
        return;
    }
}

// ---------- GO: play best move, but avoid repetition only if a non-rep alternative > draw exists ----------
// ---------- GO: correct side-to-move perspective handling ----------
void go() {
    constexpr int SEARCH_DEPTH = 4;
    Utils::displayBoard(&g_board);

    SEARCH::SearchResult result = SEARCH::g_searcher.findBestMove(g_board, SEARCH_DEPTH);
    SEARCH::printSearchResult(result);

    // Expectation: result.root_move_scores contains (CMove, int) for all legal root moves.
    if (!result.root_move_scores.empty()) {

        // --- 1) choose the best root move from the perspective of the side to move ---
        CMove best_mv = result.root_move_scores.front().first;
        int best_score_for_side = INT_MIN;
        {
            for (const auto &p : result.root_move_scores) {
                const CMove &cand = p.first;
                int raw = p.second;
                int score_for_side = (g_board.chance == WHITE) ? raw : -raw;
                if (score_for_side > best_score_for_side) {
                    best_score_for_side = score_for_side;
                    best_mv = cand;
                }
            }
        }

        // --- 2) check whether best_mv causes a third-occurrence repetition ---
        HexaBitBoardPosition tmp_best = g_board;
        uint64 dummy = 0ULL;
        if (tmp_best.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&tmp_best, dummy, best_mv);
        else MoveGeneratorBitboard::makeMove<BLACK, false>(&tmp_best, dummy, best_mv);

        HexaBitBoardPosition saved = g_board;
        g_board = tmp_best;
        std::string key_after_best = repetition_key_from_fen(get_fen_from_gboard());
        g_board = saved;

        bool best_causes_third = (count_key_occurrences(key_after_best) >= 2);

        if (!best_causes_third) {
            // Best-for-side is safe -> play it
            char buf[16] = {0};
            SEARCH::moveToString(best_mv, buf);
            uint64 apply_hash = 0ULL;
            if (g_board.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&g_board, apply_hash, best_mv);
            else MoveGeneratorBitboard::makeMove<BLACK, false>(&g_board, apply_hash, best_mv);
            push_current_position_to_history();
            printf("bestmove %s\n", buf);
            return;
        }

        // --- 3) best-for-side repeats -> search for best alternative (avoid repetition & > draw) ---
        bool found_alt = false;
        CMove chosen_alt;
        std::string chosen_uci;
        int chosen_score_for_side = INT_MIN;

        for (const auto &p : result.root_move_scores) {
            const CMove &cand = p.first;
            int raw = p.second;
            // skip null moves
            if (cand.getFrom() == 0 && cand.getTo() == 0) continue;
            // skip identical to best_mv
            if (cand.getFrom() == best_mv.getFrom() && cand.getTo() == best_mv.getTo() && cand.getFlags() == best_mv.getFlags()) continue;

            int score_for_side = (g_board.chance == WHITE) ? raw : -raw;
            if (score_for_side <= 0) continue; // must be strictly better than draw

            // simulate candidate to test repetition
            HexaBitBoardPosition tmp = g_board;
            uint64 h = 0ULL;
            if (tmp.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&tmp, h, cand);
            else MoveGeneratorBitboard::makeMove<BLACK, false>(&tmp, h, cand);

            std::string key_after = repetition_key_from_fen(get_fen_from_hexa(tmp)); // use get_fen_from_hexa for tmp

            if (count_key_occurrences(key_after) < 2) {
                // avoids repetition and > draw for side-to-move
                if (score_for_side > chosen_score_for_side) {
                    chosen_score_for_side = score_for_side;
                    chosen_alt = cand;
                    char buf[16] = {0};
                    SEARCH::moveToString(cand, buf);
                    chosen_uci = std::string(buf);
                    found_alt = true;
                }
            }
        }

        if (found_alt) {
            uint64 apply_hash = 0ULL;
            if (g_board.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&g_board, apply_hash, chosen_alt);
            else MoveGeneratorBitboard::makeMove<BLACK, false>(&g_board, apply_hash, chosen_alt);
            push_current_position_to_history();
            printf("info string Chose alternative %s to avoid repetition (score_for_side=%d)\n", chosen_uci.c_str(), chosen_score_for_side);
            printf("bestmove %s\n", chosen_uci.c_str());
            return;
        } else {
            // No alternative better than draw found -> accept repetition; play best-for-side (which is still the searcher's best-for-side)
            char buf[16] = {0};
            SEARCH::moveToString(best_mv, buf);
            uint64 apply_hash = 0ULL;
            if (g_board.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&g_board, apply_hash, best_mv);
            else MoveGeneratorBitboard::makeMove<BLACK, false>(&g_board, apply_hash, best_mv);
            push_current_position_to_history();
            printf("info string No non-rep move > draw; accepting repetition and playing best move %s\n", buf);
            printf("bestmove %s\n", buf);
            return;
        }
    }

    // Defensive fallback (if root_move_scores is empty)
    if (!result.principal_variation.empty()) {
        CMove bm = result.best_move;
        char buf[16] = {0};
        SEARCH::moveToString(bm, buf);
        uint64 apply_hash = 0ULL;
        if (g_board.chance == WHITE) MoveGeneratorBitboard::makeMove<WHITE, false>(&g_board, apply_hash, bm);
        else MoveGeneratorBitboard::makeMove<BLACK, false>(&g_board, apply_hash, bm);
        push_current_position_to_history();
        printf("bestmove %s\n", buf);
        return;
    }

    // No move -> null
    printf("bestmove 0000\n");
}

// ---------- Main UCI loop ----------
void loop(int argc, char** argv) {
    // New: Parse command-line arguments for network name
    if (argc > 1) {
        std::string net_base_name = argv[1];
        g_nnue_model_path = "nets/" + net_base_name + ".bin";
        g_nnue_mapping_path = "nets/" + net_base_name + "_mapping.bin";
        printf("info string Using custom network: %s\n", net_base_name.c_str());
    } else {
        printf("info string Using default compiled network\n");
    }

    newgame();
    SEARCH::g_searcher.init();

    std::string token, cmd;
    while (true) {
        if (!std::getline(std::cin, cmd)) break;
        std::istringstream is(cmd);
        token.clear();
        is >> std::skipws >> token;
        if (token == "quit" || token == "stop") break;
        else if (token == "ucinewgame") {
            newgame();
            printf("info string newgame\n");
        }
        else if (token == "position") {
            handle_position(is);
        }
        else if (token == "move") {
            std::string mv;
            if (is >> mv) {
                bool ok = apply_move_internal(mv);
                if (ok) {
                    printf("info string Applied move %s\n", mv.c_str());
                    Utils::displayBoard(&g_board);
                } else {
                    printf("info string Illegal move %s\n", mv.c_str());
                }
            } else {
                printf("info string No move provided\n");
            }
        }
        else if (token == "go") {
            go();
        }
        else if (token == "uci") {
            printf("id name NNUE32-CUDA\n");
            printf("id author You\n");
            printf("uciok\n");
        }
        else if (token == "isready") {
            printf("readyok\n");
        }
        else {
            printf("info string Unknown command: %s\n", cmd.c_str());
        }
    }

    SEARCH::g_searcher.cleanup();
}

} // namespace UCI
