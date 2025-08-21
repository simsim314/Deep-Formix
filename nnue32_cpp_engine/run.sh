#!/bin/bash
# This script creates and runs a Python script to play your C++ engine
# against Stockfish in a fair, automated match.

# Create the Python match script
cat << 'EOF' > play_match.py
#!/usr/bin/env python3
"""
play_match.py: Plays a UCI vs UCI match between a custom engine and Stockfish.
This script orchestrates the match, manages games from a list of FENs,
alternates colors, and saves the results to a PGN file.
"""

import sys
import os
import chess
import chess.engine
import chess.pgn
from datetime import datetime
from pathlib import Path

# ============================================================================
#  MATCH CONFIGURATION
# ============================================================================

# --- Engine Paths ---
# Your compiled C++ engine, must be in the same directory and executable
NNUE_ENGINE_PATH = "./nnue32_engine"
# Path to your Stockfish executable
STOCKFISH_PATH   = "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"

# --- Match Parameters ---
# Set the thinking depth for both engines. This is crucial for a fair match.
MATCH_DEPTH = 1

# --- PGN Output ---
SAVE_PGN = True
PGN_FILE_PREFIX = "cpp_nnue_vs_stockfish"

# --- Opening Positions ---
# A list of (FEN, Name) tuples to start games from.
POSITIONS = [
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian Game"),
    ("r1bqkbnr/pp1ppppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 4 4", "Sicilian Defense"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 4 5", "Queen's Gambit Declined"),
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Ruy Lopez"),
    ("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3", "French Defense"),
    ("r1bqkb1r/pp1npppp/2p2n2/8/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 1 6", "Caro-Kann"),
    ("r1bqk1nr/pppp1pbp/2n3p1/4p3/2P5/2N3P1/PP1PPP1P/R1BQKBNR w KQkq - 0 5", "English Opening"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 0 7", "King's Indian Defense"),
    ("r1bqk2r/pp3ppp/2nppn2/2p5/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 8", "Nimzo-Indian"),
    ("rnbqkb1r/ppp1pppp/5n2/3P4/8/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 4", "Scandinavian Defense"),
]

# ============================================================================
#  Match Orchestration Class
# ============================================================================

class Match:
    def __init__(self, nnue_path, stockfish_path, depth):
        self.nnue_path = nnue_path
        self.stockfish_path = stockfish_path
        self.limit = chess.engine.Limit(depth=depth)
        self.results = {"nnue_wins": 0, "stockfish_wins": 0, "draws": 0, "errors": 0}

    def play_game(self, fen, position_name, nnue_is_white):
        """
        Plays one game between the two engines from a given FEN.
        Returns the completed chess.pgn.Game object.
        """
        board = chess.Board(fen)
        game = chess.pgn.Game()
        game.headers["Event"] = f"NNUE-32 vs Stockfish (Depth {self.limit.depth})"
        game.headers["Site"] = "Local Machine"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = position_name
        game.headers["White"] = "NNUE-32 (C++)" if nnue_is_white else "Stockfish"
        game.headers["Black"] = "Stockfish" if nnue_is_white else "NNUE-32 (C++)"
        game.headers["FEN"] = fen
        game.setup(board)
        node = game

        print(f"--- Starting: {position_name} ({game.headers['White']} vs {game.headers['Black']}) ---")

        try:
            # Open fresh engines for each game to ensure a clean state
            with chess.engine.SimpleEngine.popen_uci(self.nnue_path) as nnue_engine, \
                 chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as stockfish_engine:

                while not board.is_game_over(claim_draw=True):
                    if (board.turn == chess.WHITE) == nnue_is_white:
                        engine_name = "NNUE"
                        result = nnue_engine.play(board, self.limit)
                    else:
                        engine_name = "Stockfish"
                        result = stockfish_engine.play(board, self.limit)

                    if result.move is None:
                        print(f"Error: {engine_name} returned a null move. Game aborted.", file=sys.stderr)
                        game.headers["Termination"] = "Aborted"
                        self.results["errors"] += 1
                        break
                    
                    print(f"  {board.fullmove_number}.{' ' if board.turn == chess.WHITE else '..'} {result.move.uci():<6} ({engine_name})", flush=True)

                    board.push(result.move)
                    node = node.add_variation(result.move)

        except Exception as e:
            print(f"\nFATAL ERROR during game: {e}", file=sys.stderr)
            game.headers["Termination"] = "Error"
            self.results["errors"] += 1
            return game

        result_str = board.result(claim_draw=True)
        game.headers["Result"] = result_str
        print(f"Game Over. Result: {result_str}")

        # Update scores
        if result_str == "1-0":
            self.results["nnue_wins" if nnue_is_white else "stockfish_wins"] += 1
        elif result_str == "0-1":
            self.results["stockfish_wins" if nnue_is_white else "nnue_wins"] += 1
        elif result_str == "1/2-1/2":
            self.results["draws"] += 1
            
        return game

    def run_match(self):
        all_games = []
        total_games = len(POSITIONS) * 2
        
        for i, (fen, name) in enumerate(POSITIONS):
            # Game 1: NNUE plays White
            print(f"\n>>> Game {i*2 + 1}/{total_games} <<<")
            game1 = self.play_game(fen, name, nnue_is_white=True)
            if game1: all_games.append(game1)
            
            # Game 2: NNUE plays Black
            print(f"\n>>> Game {i*2 + 2}/{total_games} <<<")
            game2 = self.play_game(fen, name, nnue_is_white=False)
            if game2: all_games.append(game2)

        # Save PGN file
        if SAVE_PGN and all_games:
            pgn_filename = f"{PGN_FILE_PREFIX}_d{self.limit.depth}_{datetime.now().strftime('%Y%m%d_%H%M')}.pgn"
            with open(pgn_filename, "w", encoding="utf-8") as f:
                for game in all_games:
                    print(game, file=f, end="\n\n")
            print(f"\n[SUCCESS] All {len(all_games)} games saved to '{pgn_filename}'.")

        # Print final results
        print("\n" + "="*40)
        print("           FINAL MATCH RESULTS")
        print("="*40)
        print(f"NNUE-32 Wins:     {self.results['nnue_wins']}")
        print(f"Stockfish Wins:   {self.results['stockfish_wins']}")
        print(f"Draws:            {self.results['draws']}")
        if self.results['errors'] > 0:
            print(f"Errors/Aborts:    {self.results['errors']}")
        print("-"*40)
        
        total_played = self.results['nnue_wins'] + self.results['stockfish_wins'] + self.results['draws']
        if total_played > 0:
            score = self.results['nnue_wins'] + 0.5 * self.results['draws']
            percentage = (score / total_played) * 100
            print(f"NNUE-32 Score:    {score}/{total_played} ({percentage:.1f}%)")
        print("="*40)

def main():
    # Verify that the engine executables exist and are executable
    for path in [NNUE_ENGINE_PATH, STOCKFISH_PATH]:
        if not Path(path).exists():
            print(f"Error: Required engine file not found: '{path}'", file=sys.stderr)
            sys.exit(1)
        if not os.access(path, os.X_OK):
            print(f"Error: Engine file not executable: '{path}'. Please run: chmod +x {path}", file=sys.stderr)
            sys.exit(1)
    
    # Create and run the match
    match = Match(NNUE_ENGINE_PATH, STOCKFISH_PATH, MATCH_DEPTH)
    match.run_match()

if __name__ == "__main__":
    main()
EOF

# Make the generated script executable and run it
chmod +x play_match.py
echo "-----> Python match script 'play_match.py' has been created."
echo "-----> Now executing the match..."
python3 play_match.py
