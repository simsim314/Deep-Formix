#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <string>
#include <vector>
#include <iostream>
#include "chess.h"
#include "search.cuh" // For access to the SearchResult struct

// A struct to hold all the data for a single test case
struct TestCase {
    std::string name;
    std::string fen;
    int depth;
    std::string expected_move_uci;
    int expected_score;
};

// Global counters for the test summary
int tests_passed = 0;
int tests_failed = 0;

// The core test function to check a result and print PASS/FAIL
void check_result(const std::string& test_name, bool condition, const std::string& failure_details) {
    if (condition) {
        std::cout << "[PASS] " << test_name << std::endl;
        tests_passed++;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
        std::cout << "       " << failure_details << std::endl;
        tests_failed++;
    }
}

#endif // TEST_RUNNER_H
