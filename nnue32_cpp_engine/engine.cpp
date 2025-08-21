#include "estimator.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <ostream>

// ---------- FEN accessor detection & helper ----------

// trait checks for various possible FEN accessor names
template<typename T, typename = void> struct has_fen_method : std::false_type {};
template<typename T> struct has_fen_method<T, std::void_t<decltype(std::declval<const T>().fen())>> : std::true_type {};

template<typename T, typename = void> struct has_getFen_method : std::false_type {};
template<typename T> struct has_getFen_method<T, std::void_t<decltype(std::declval<const T>().getFen())>> : std::true_type {};

template<typename T, typename = void> struct has_toFen_method : std::false_type {};
template<typename T> struct has_toFen_method<T, std::void_t<decltype(std::declval<const T>().toFen())>> : std::true_type {};

template<typename T, typename = void> struct has_to_fen_method : std::false_type {};
template<typename T> struct has_to_fen_method<T, std::void_t<decltype(std::declval<const T>().to_fen())>> : std::true_type {};

// Generic get_fen that picks the available accessor at compile time.
// If none of the common accessors exist, falls back to operator<< (may not be FEN).
template<typename BoardT>
std::string get_fen(const BoardT& b) {
    if constexpr (has_fen_method<BoardT>::value) {
        return b.fen();
    } else if constexpr (has_getFen_method<BoardT>::value) {
        return b.getFen();
    } else if constexpr (has_toFen_method<BoardT>::value) {
        return b.toFen();
    } else if constexpr (has_to_fen_method<BoardT>::value) {
        return b.to_fen();
    } else {
        // Fallback: try streaming the board. NOTE: operator<< may not produce RFC FEN.
        std::ostringstream oss;
        oss << b;
        return oss.str();
    }
}

// ---------- Utility: repetition key from FEN ----------
// Extract the repetition-relevant key from a FEN string:
// piece placement, side to move, castling rights, en-passant square.
static inline std::string repetition_key_from_fen(const std::string& fen) {
    std::istringstream iss(fen);
    std::string part;
    std::string key;
    for (int i = 0; i < 4; ++i) {
        if (!(iss >> part)) break;
        if (i) key += " ";
        key += part;
    }
    return key;
}

// ---------- score -> probability helper (optional) ----------
static inline float cp_to_prob(float cp) {
    const float scale = 400.0f;
    float x = cp / scale;
    if (x > 50.0f) x = 50.0f;
    if (x < -50.0f) x = -50.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// Forward declarations (assumed to be defined/used elsewhere or in estimator.hpp)
float gelu(float x);
template<int N>
void layer_norm(Eigen::Vector<float, N>& x, const Eigen::VectorXf& weight, const Eigen::VectorXf& bias);

// ---------- run_test (unchanged logic, but uses get_fen if needed) ----------
void run_test(NNUE_Estimator& estimator, const std::string& fen) {
    std::cout << std::fixed << std::setprecision(6);
    chess::Board board(fen);
    int bucket_idx = estimator.get_descriptor_index_public(board);
    std::cout << "CPP: Using Bucket Index: " << bucket_idx << std::endl;
    const auto& network = estimator.get_network_public(bucket_idx);
    std::cout << "--- CPP: FORWARD PASS ---" << std::endl;
    Eigen::Vector<float, 2240> x_emb;
    estimator.compute_x_emb_public(x_emb, board);
    std::cout << "CPP: SUM(x_emb)            = " << x_emb.sum() << std::endl;
    Eigen::Vector<float, 256> x1_pre = network.fc1_w * x_emb + network.fc1_b;
    std::cout << "CPP: SUM(fc1 output)       = " << x1_pre.sum() << std::endl;
    Eigen::Vector<float, 256> x1 = x1_pre.unaryExpr(&gelu);
    layer_norm<256>(x1, network.ln1_w, network.ln1_b);
    std::cout << "CPP: SUM(after gelu+ln1)   = " << x1.sum() << std::endl;
    Eigen::Vector<float, 32> x2_pre = network.fc2_w * x1 + network.fc2_b;
    std::cout << "CPP: SUM(fc2 output)       = " << x2_pre.sum() << std::endl;
    Eigen::Vector<float, 32> x2 = x2_pre.unaryExpr(&gelu);
    layer_norm<32>(x2, network.ln2_w, network.ln2_b);
    std::cout << "CPP: SUM(after gelu+ln2)   = " << x2.sum() << std::endl;
    for (int i = 0; i < 12; ++i) {
        network.blocks[i].forward(x2);
        std::cout << "CPP: SUM(after block " << std::setw(2) << i << ")   = " << x2.sum() << std::endl;
    }
    x2 = x2.unaryExpr(&gelu);
    Eigen::Vector<float, 51> logits = network.fc_out_w * x2 + network.fc_out_b;
    std::cout << "CPP: SUM(logits)           = " << logits.sum() << std::endl;
    logits = (logits.array() - logits.maxCoeff()).exp();
    logits /= logits.sum();
    float p_win = logits.dot(estimator.get_bins_public());
    std::cout << "CPP: FINAL p(win)          = " << p_win << std::endl;
}

// ---------- UCI loop with repetition tracking & avoidance ----------
void uci_loop(NNUE_Estimator& estimator) {
    chess::Board board;
    std::string line;

    // history of repetition keys (the relevant 4 FEN fields) in chronological order.
    // Always keep the current position's key as the last element.
    std::vector<std::string> position_history;

    auto push_current_position_to_history = [&](const chess::Board& b) {
        std::string fen;
        try {
            fen = get_fen(b);
        } catch (...) {
            fen = "";
        }
        position_history.push_back(repetition_key_from_fen(fen));
    };

    auto rebuild_history_from_position = [&](const std::string& fen_start, const std::vector<chess::Move>& moves_applied) {
        position_history.clear();
        std::string start_key = repetition_key_from_fen(fen_start);
        position_history.push_back(start_key);

        chess::Board tmp(fen_start);
        for (const auto& m : moves_applied) {
            tmp.makeMove(m);
            std::string f;
            try {
                f = get_fen(tmp);
            } catch (...) {
                f = "";
            }
            position_history.push_back(repetition_key_from_fen(f));
        }
    };

    auto count_key_occurrences = [&](const std::string& key) {
        return std::count(position_history.begin(), position_history.end(), key);
    };

    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "uci") {
            std::cout << "id name NNUE32-CPP-Debug Engine" << std::endl;
            std::cout << "id author You" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "ucinewgame") {
            board.setFen(chess::constants::STARTPOS);
            position_history.clear();
            try {
                position_history.push_back(repetition_key_from_fen(get_fen(board)));
            } catch (...) {
                position_history.push_back("");
            }
        } else if (token == "position") {
            std::string sub_token;
            std::string fen;
            iss >> sub_token;
            // We'll collect moves in their UCI string form and apply after setting fen
            std::vector<std::string> uci_moves;

            if (sub_token == "startpos") {
                fen = chess::constants::STARTPOS;
                // Now read optional "moves ..." from the rest of the stream
                std::string t;
                while (iss >> t) {
                    if (t == "moves") {
                        std::string mv;
                        while (iss >> mv) uci_moves.push_back(mv);
                        break;
                    }
                }
            } else if (sub_token == "fen") {
                // read all fen fields (up to 6) until we hit "moves" or end
                std::vector<std::string> fen_parts;
                std::string field;
                while (iss >> field) {
                    if (field == "moves") break;
                    fen_parts.push_back(field);
                }
                std::ostringstream of;
                for (size_t i = 0; i < fen_parts.size(); ++i) {
                    if (i) of << " ";
                    of << fen_parts[i];
                }
                fen = of.str();

                // If we stopped at "moves" there might be moves remaining in iss
                std::string mv;
                while (iss >> mv) uci_moves.push_back(mv);
            } else {
                // Unexpected token: treat rest of the line as fen (robust fallback)
                std::ostringstream of;
                of << sub_token;
                std::string rest;
                while (iss >> rest) {
                    of << " " << rest;
                }
                fen = of.str();
            }

            // Set the board and apply moves one-by-one, updating position_history
            board.setFen(fen);
            position_history.clear();
            try {
                position_history.push_back(repetition_key_from_fen(get_fen(board)));
            } catch (...) {
                position_history.push_back("");
            }

            for (const auto& u : uci_moves) {
                chess::Move m = chess::uci::uciToMove(board, u);
                board.makeMove(m);
                try {
                    position_history.push_back(repetition_key_from_fen(get_fen(board)));
                } catch (...) {
                    position_history.push_back("");
                }
            }
        } else if (token == "go") {
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, board);
            if (moves.empty()) {
                std::cout << "bestmove (none)" << std::endl;
                continue;
            }
            std::cout << "info string Analyzing " << moves.size() << " legal moves." << std::endl;

            struct Candidate {
                chess::Move move;
                float score_cp;
                float p_win;
                bool causes_repetition;
                std::vector<float> layer_sums;
            };

            std::vector<Candidate> repetition_moves;
            std::vector<Candidate> non_repetition_moves;

            for (const auto& move : moves) {
                chess::Board temp_board = board;
                temp_board.makeMove(move);

                std::string fen_after;
                try {
                    fen_after = get_fen(temp_board);
                } catch (...) {
                    fen_after = "";
                }
                std::string key_after = repetition_key_from_fen(fen_after);
                int occurrences = count_key_occurrences(key_after);
                bool would_cause_third = (occurrences >= 2);

                // ---- FIXED: ensure we use the same sign convention for printing and selection ----
                DetailedEvaluation detailed_eval = estimator.evaluate_detailed(temp_board);

                // original code used -detailed_eval.score for selection/printing.
                // Keep that convention: positive score_root = better for side that made the move.
                float score_root = -detailed_eval.score;
                float p_win = cp_to_prob(score_root);

                Candidate c;
                c.move = move;
                c.score_cp = score_root;       // store the score in the original polarity used for decisions
                c.p_win = p_win;
                c.causes_repetition = would_cause_third;
                c.layer_sums = detailed_eval.layer_sums;
                // -------------------------------------------------------------------------------

                if (would_cause_third) repetition_moves.push_back(c);
                else non_repetition_moves.push_back(c);

                // Print using the same convention (score_root equals -detailed_eval.score)
                std::cout << "info string move " << chess::uci::moveToUci(move)
                          << " score cp " << score_root
                          << " p_win " << std::fixed << std::setprecision(6) << p_win
                          << (would_cause_third ? " (repetition-risk)" : "")
                          << " layers";
                for (float sum : detailed_eval.layer_sums) {
                    std::cout << " " << std::fixed << std::setprecision(6) << sum;
                }
                std::cout << std::endl;
            }

            chess::Move chosen_move;
            bool chosen_is_set = false;

            if (!repetition_moves.empty()) {
                // find best non-repetition alternative by p_win
                if (!non_repetition_moves.empty()) {
                    auto best_nonrep = std::max_element(non_repetition_moves.begin(), non_repetition_moves.end(),
                                                       [](const Candidate& a, const Candidate& b){
                                                           return a.p_win < b.p_win;
                                                       });
                    if (best_nonrep != non_repetition_moves.end() && best_nonrep->p_win > 0.5f) {
                        chosen_move = best_nonrep->move;
                        chosen_is_set = true;
                        std::cout << "info string Avoiding threefold repetition: choosing alternative with p_win=" << std::fixed << std::setprecision(6) << best_nonrep->p_win << std::endl;
                    }
                }
                if (!chosen_is_set) {
                    // accept draw: choose best repetition move (by score_cp, fallback p_win)
                    auto best_rep = std::max_element(repetition_moves.begin(), repetition_moves.end(),
                                                     [](const Candidate& a, const Candidate& b){
                                                         if (a.score_cp == b.score_cp) return a.p_win < b.p_win;
                                                         return a.score_cp < b.score_cp;
                                                     });
                    if (best_rep != repetition_moves.end()) {
                        chosen_move = best_rep->move;
                        chosen_is_set = true;
                        std::cout << "info string No alternative with p_win>0.5; accepting draw via repetition (playing " << chess::uci::moveToUci(chosen_move) << ")." << std::endl;
                    }
                }
            }

            if (!chosen_is_set) {
                // choose best among all moves by score_cp (fallback to p_win)
                std::vector<Candidate> all_candidates;
                all_candidates.reserve(non_repetition_moves.size() + repetition_moves.size());
                all_candidates.insert(all_candidates.end(), non_repetition_moves.begin(), non_repetition_moves.end());
                all_candidates.insert(all_candidates.end(), repetition_moves.begin(), repetition_moves.end());
                if (!all_candidates.empty()) {
                    auto best = std::max_element(all_candidates.begin(), all_candidates.end(),
                                                 [](const Candidate& a, const Candidate& b){
                                                     if (a.score_cp == b.score_cp) return a.p_win < b.p_win;
                                                     return a.score_cp < b.score_cp;
                                                 });
                    chosen_move = best->move;
                    chosen_is_set = true;
                }
            }

            if (!chosen_is_set) {
                chosen_move = moves[0];
            }

            // apply chosen move and update history
            board.makeMove(chosen_move);
            try {
                position_history.push_back(repetition_key_from_fen(get_fen(board)));
            } catch (...) {
                position_history.push_back("");
            }

            std::cout << "bestmove " << chess::uci::moveToUci(chosen_move) << std::endl;
        } else if (token == "quit") {
            break;
        }
    }
}

// ---------- main ----------
int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "error: requires NET_NAME (.bin file) as first argument." << std::endl;
            return 1;
        }
        std::string net_name = argv[1];
        std::string mapping_name = net_name.substr(0, net_name.find_last_of('.')) + "_mapping.bin";

        NNUE_Estimator estimator(net_name, mapping_name);
        if (argc > 2 && std::string(argv[2]) == "test") {
            if (argc > 3) {
                run_test(estimator, std::string(argv[3]));
            } else {
                std::cerr << "error: test mode requires a FEN string." << std::endl;
                return 1;
            }
        } else {
            uci_loop(estimator);
        }
    } catch (const std::exception& e) {
        std::cerr << "error " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
