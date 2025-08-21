#!/usr/bin/env python3
"""
positions_analyzer.py - A standalone script to analyze positions from the dataset.

This script directly reads .bag files to analyze a sample of positions and
prints a statistical analysis based on queen presence, piece count, and pawn count.
"""

import chess
import bagz
from collections import Counter
from pathlib import Path

# --- IMPORTANT: Import the decoder from your project's constants file ---
try:
    from constants import CODERS
except ImportError:
    print("Error: Could not import CODERS from constants.py.")
    print("Please make sure this script is in the same directory as your project files.")
    exit()

# --- Configuration ---
WORK_DIR = 'chess_pipeline'
POSITIONS_TO_ANALYZE = 10000

def analyze_position_features():
    """
    Directly reads .bag files and prints a statistical analysis of position features.
    """
    work_dir = Path(WORK_DIR)
    if not work_dir.exists():
        print(f"Error: Work directory '{WORK_DIR}' not found.")
        return

    all_bag_files = sorted(list(work_dir.glob("*.bag")))
    if not all_bag_files:
        print(f"No .bag files found in '{WORK_DIR}'. Nothing to analyze.")
        return
        
    print(f"Found {len(all_bag_files)} .bag files to analyze.")

    # A dictionary to store the counts of each unique configuration.
    # The key will be a tuple: (has_queen, piece_count, pawn_count)
    configuration_counts = Counter()
    total_positions_processed = 0
    
    print(f"\nAnalyzing up to {POSITIONS_TO_ANALYZE} positions...")

    try:
        for bag_path in all_bag_files:
            if total_positions_processed >= POSITIONS_TO_ANALYZE:
                print("Reached analysis limit.")
                break
            
            print(f"Reading from: {bag_path.name}...")
            
            try:
                reader = bagz.BagFileReader(str(bag_path))
                
                for i in range(len(reader)):
                    if total_positions_processed >= POSITIONS_TO_ANALYZE:
                        break
                        
                    fen, _, _ = CODERS['action_value'].decode(reader[i])
                    board = chess.Board(fen)
                    
                    # --- FEATURE CALCULATION ---
                    has_queen = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))

                    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
                    knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
                    bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
                    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
                    piece_count = rooks + knights + bishops + queens

                    pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
                    
                    configuration_key = (has_queen, piece_count, pawn_count)
                    
                    configuration_counts[configuration_key] += 1
                    total_positions_processed += 1
                    
            except Exception as e:
                print(f"  - Could not read file {bag_path.name}. Error: {e}")
                continue

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        
    if total_positions_processed == 0:
        print("\nCould not read any positions from the dataset.")
        return
        
    print("\n\n--- Chess Position Feature Analysis ---")
    print(f"\nTotal Positions Analyzed: {total_positions_processed}\n")
    
    # --- MODIFICATION START ---
    # Sort by a tuple: (has_queen, piece_count, pawn_count) in descending order.
    # This groups by queen presence, then by piece count, then by pawn count.
    sorted_configurations = sorted(
        configuration_counts.items(), 
        key=lambda item: (item[0][0], item[0][1], item[0][2]), 
        reverse=True
    )
    # --- MODIFICATION END ---
    
    print(f"{'Queen?':<12} | {'# Pieces':<10} | {'# Pawns':<10} | {'Count':<15} | {'Percentage':<10}")
    print("-" * 70)

    for config, count in sorted_configurations:
        has_queen_bool, piece_count, pawn_count = config
        queen_status = "Has Queen" if has_queen_bool else "No Queen"
        percentage = (count / total_positions_processed) * 100
        
        print(f"{queen_status:<12} | {piece_count:<10} | {pawn_count:<10} | {count:<15} | {percentage:9.2f}%")


if __name__ == "__main__":
    analyze_position_features()
