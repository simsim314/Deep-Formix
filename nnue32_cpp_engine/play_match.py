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
SAVE_PGN = False
PGN_FILE_PREFIX = "cpp_nnue_vs_stockfish"

# --- Opening Positions ---
# A list of (FEN, Name) tuples to start games from.
POSITIONS = [
    # Classical and well-known
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3", "Italian Game"),
    ("rnbqkb1r/pp1ppppp/2p2n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 3", "Sicilian Defense (Najdorf setup)"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 3", "Queen's Gambit Declined"),
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3", "Ruy Lopez"),
    ("rnbqkb1r/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "French Defense"),
    ("r1bqkb1r/pp1npppp/2p2n2/8/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 2 3", "Caro-Kann Defense"),
    ("r1bqk1nr/pppp1pbp/2n3p1/4p3/2P5/2N3P1/PP1PPP1P/R1BQKBNR w KQkq - 0 4", "English Opening"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 0 6", "King's Indian Defense"),
    ("r1bqk2r/pp3ppp/2nppn2/2p5/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 7", "Nimzo-Indian Defense"),
    ("rnbqkb1r/ppp1pppp/5n2/3P4/8/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "Scandinavian Defense"),

    # Gambits and sharp lines
    ("rnbqkb1r/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3", "King's Gambit (after 1.e4 e5 2.f4)"),
    ("rnbqkbnr/ppp2ppp/8/3pp3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Budapest Gambit"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Albin Counter-Gambit"),
    ("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Center Game"),
    ("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Danish Gambit (after 1.e4 e5 2.d4 exd4)"),
    ("rnbqkbnr/pppp1ppp/8/4p3/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 2", "Vienna Game"),
    ("rnbqkbnr/ppp1pppp/8/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 1 2", "Queen's Gambit Accepted"),

    # Hypermodern and flank openings
    ("rnbqkb1r/p1pppppp/1p6/8/3NP3/8/PPP1PPPP/RNBQKB1R w KQkq - 0 2", "Benko Gambit"),
    ("rnbqkbnr/pp1ppppp/2p5/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", "Englund Gambit"),
    ("rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Tarrasch Defense"),
    ("rnbqkbnr/pppp1ppp/8/8/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 2", "London System"),
    ("rnbqkb1r/ppp2ppp/3p1n2/3Pp3/2P5/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 3", "Dutch Defense"),
    ("rnbqkbnr/pp1pp1pp/2p2p2/8/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 0 2", "Benoni Defense"),
    ("rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "King's Indian Attack"),
    ("rnbqkbnr/pppppppp/8/8/3PP1P1/8/PPP2P1P/RNBQKBNR w KQkq - 0 2", "Bird's Opening"),

    # Esoteric and unusual systems
    ("rnbqkbnr/pppppp1p/8/6p1/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Grob's Attack"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 2", "Nimzowitsch-Larsen Attack"),
    ("rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Polish Opening (Sokolsky)"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Zukertort Opening"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Jobava London"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Colle System"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Bongcloud Attack"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Wayward Queen Attack"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Ponziani Opening"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Torre Attack"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Réti Opening"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Polish Opening"),
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Elephant Gambit"),
    
    ("rnbqkbnr/pp2pppp/8/2pp4/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 3", "Trompowsky Attack"),
    ("rnbqkbnr/ppp1pppp/8/3p4/2BPP3/8/PPP2PPP/RNBQK1NR w KQkq - 0 3", "Evans Gambit"),
    ("rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2", "Sicilian Alapin"),
    ("rnbqkbnr/ppp2ppp/4p3/3p4/2PPP3/8/PP3PPP/RNBQKBNR w KQkq - 0 3", "Slav Defense"),
    ("rnbqkb1r/pp2pppp/5n2/2pp4/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 3", "Grünfeld Defense"),
    ("rnbqkbnr/ppp1pppp/8/3p4/2BPP3/8/PPP2PPP/RNBQK1NR w KQkq - 0 3", "Scotch Game"),
    ("rnbqkbnr/ppp1pppp/8/3p4/3PP1P1/8/PPP2P1P/RNBQKBNR w KQkq - 0 3", "Bird's Opening From Gambit"),
    ("rnbqkbnr/pppp1ppp/8/4p3/2BPP3/8/PPP2PPP/RNBQK1NR w KQkq - 0 3", "Bishop's Opening"),
    ("rnbqkbnr/ppp1pppp/8/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 3", "Ponziani Opening"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/3PP3/2P2N2/PP3PPP/RNBQKB1R w KQkq - 0 3", "Colle-Zukertort System"),
    ("rnbqkbnr/ppp1pppp/8/3p4/3PP3/3B4/PPP2PPP/RNBQK1NR w KQkq - 0 3", "Torre Attack (d4 & Bg5)"),
    ("rnbqkbnr/pp1ppppp/8/2p5/2B1P3/8/PPP2PPP/RNBQK1NR w KQkq - 0 2", "Latvian Gambit"),

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
        #if SAVE_PGN and all_games:
            #pgn_filename = f"{PGN_FILE_PREFIX}_d{self.limit.depth}_{datetime.now().strftime('%Y%m%d_%H%M')}.pgn"
            #with open(pgn_filename, "w", encoding="utf-8") as f:
            #    for game in all_games:
            #        print(game, file=f, end="\n\n")
            #print(f"\n[SUCCESS] All {len(all_games)} games saved to '{pgn_filename}'.")

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
