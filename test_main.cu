#include "test_runner.h"
#include "MoveGeneratorBitboard.h"
#include "search.cuh"
#include "utils.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

// Function to define the suite of tests to run
std::vector<TestCase> get_test_suite() {
    return {
        // --- Static Evaluation Tests (Depth 0) ---
        {
            "Static Eval: Initial Position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0, "", 11300
        },

        // --- Move Selection Tests (Depth 1) ---
        {
            "Move Selection: Simple Capture",
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            1, "f6e4", 100
        },
        {
            "Move Selection: Mate-in-1",
            "4k3/R7/8/8/8/8/8/4K3 w - - 0 1",
            1, "a7a8", 100000
        },

        // --- Minimax Functionality Tests (Depth 2) ---
        {
            "Minimax: Setup a Fork",
            "1k1r4/pp1b1p2/2p1p3/3p3p/1P2P3/2P5/P1Q2P1q/2K1R3 w - - 0 28",
            2, "e4d5", 0
        },
        {
            "Minimax: Avoid a Blunder",
            "rnb1kbnr/pppp1ppp/8/4p3/5q2/6P1/PPPPP2P/RNBQKBNR w KQkq - 0 4",
            2, "g3f4", 0
        },

        // --- New Deep Search Test Positions ---
        {
            "Depth 5: Starting Position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            5, "", 0 // Placeholder move and score
        },
        {
            "Depth 3: Tactical Test 1",
            "3r3k/pp3p1n/2p5/7q/2P3R1/7P/PP1rQ1P1/5R1K w - - 8 31",
            3, "", 0 // Placeholder move and score
        },
        {
            "Depth 5: Tactical Test 2",
            "6rk/pp3Q1n/2p5/8/2P5/7P/PP4r1/5R1K b - - 0 33",
            5, "", 0 // Placeholder move and score
        },
		{
			"Depth 5: O-O test",
			"rn2kb1r/pp2pppp/2p2n2/3q1b2/8/5N1P/PPPPBPP1/RNBQK2R w KQkq - 1 6",
			4, "", 0 // Placeholder move and score
		},


    };
}

int main() {
    // Initialize the engine components
    MoveGeneratorBitboard::init();
    SEARCH::init();

    auto test_suite = get_test_suite();

    std::cout << "=================================" << std::endl;
    std::cout << "  Running Chess Engine Test Suite  " << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "Found " << test_suite.size() << " tests to run..." << std::endl << std::endl;


    for (const auto& test : test_suite) {
        HexaBitBoardPosition board;
        BoardPosition pos088;
        Utils::readFENString(const_cast<char*>(test.fen.c_str()), &pos088);
        std::cout << "fen: " << test.fen << std::endl;
        Utils::board088ToHexBB(&board, &pos088);
        Utils::displayBoard(&board);

        // A depth 0 test is for static eval, but we must search to depth 1 to get the root score.
        int search_depth = (test.depth == 0) ? 1 : test.depth;
        SEARCH::SearchResult result = SEARCH::findBestMove(board, search_depth);
        
        // After search, print the result for inspection
        std::cout << "  > Engine output for '" << test.name << "':" << std::endl;
        std::cout << "    ";
        SEARCH::printSearchResult(result);
        
        bool success;
        std::string failure_details;

        if (test.depth == 0) {
            // Static eval test
            success = (result.score == test.expected_score);
            if (!success) {
                std::stringstream ss;
                ss << "Expected score: " << test.expected_score << ", Got: " << result.score;
                failure_details = ss.str();
            }
        } else {
            // Search test
            char move_buffer[6];
            SEARCH::moveToString(result.best_move, move_buffer);
            std::string found_move_uci(move_buffer);
            
            success = (found_move_uci == test.expected_move_uci); // For now, just check the move
            if (!success) {
                std::stringstream ss;
                ss << "Expected move: " << test.expected_move_uci << ", Got: " << found_move_uci;
                failure_details = ss.str();
            }
        }

        check_result(test.name, success, failure_details);
    }

    std::cout << "\n---------------------------------" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  PASSED: " << tests_passed << std::endl;
    std::cout << "  FAILED: " << tests_failed << std::endl;
    std::cout << "=================================" << std::endl;


    SEARCH::terminate();
    // Return a non-zero exit code if any tests failed
    return (tests_failed > 0) ? 1 : 0;
}
