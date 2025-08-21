#!/usr/bin/env python3
"""
play_vs_stockfish_32.py - Play the 32-bucket NNUE engine vs Stockfish.
(Corrected version with 10 positions and depth 1)
"""

import chess
import chess.engine
import chess.pgn
import sys
from datetime import datetime
from pathlib import Path

try:
    from nnue32_engine import NNUE32Engine
except ImportError:
    print("FATAL: Could not import NNUE32Engine from nnue32_engine.py.", file=sys.stderr)
    sys.exit(1)

NNUE_CHECKPOINT = "nnuew_resnet_step_01105000.pt"
STOCKFISH_PATH =  "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"
STOCKFISH_DEPTH = 1
BUCKET_MAPPING_FILE = "bucket_mapping.json"

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

class PositionalMatch:
    def __init__(self, nnue_checkpoint, stockfish_path, stockfish_depth):
        self.nnue_engine = NNUE32Engine(nnue_checkpoint, BUCKET_MAPPING_FILE)
        self.stockfish_path = stockfish_path
        self.stockfish_limit = chess.engine.Limit(depth=stockfish_depth)
        self.results = {"nnue_wins": 0, "stockfish_wins": 0, "draws": 0}

    def play_game(self, fen, position_name, nnue_is_white):
        board = chess.Board(fen)
        game = chess.pgn.Game()
        game.headers["Event"] = f"NNUE-32 vs Stockfish (Depth {self.stockfish_limit.depth})"
        game.headers["White"] = "NNUE-32" if nnue_is_white else "Stockfish"
        game.headers["Black"] = "Stockfish" if nnue_is_white else "NNUE-32"
        game.headers["FEN"] = fen
        game.setup(board)
        node = game
        print(f"\n--- Starting Game: {position_name} ({game.headers['White']} vs {game.headers['Black']}) ---")

        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as stockfish:
                while not board.is_game_over(claim_draw=True):
                    if (board.turn == chess.WHITE) == nnue_is_white:
                        move = self.nnue_engine.search_best_move(board)
                        print(f"NNUE plays {move.uci()}", end=' | ', flush=True)
                    else:
                        result = stockfish.play(board, self.stockfish_limit)
                        move = result.move
                        print(f"Stockfish plays {move.uci()}", end='\n', flush=True)
                    if move is None: break
                    board.push(move)
                    node = node.add_variation(move)
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            return None

        game.headers["Result"] = board.result(claim_draw=True)
        print(f"Game Over. Result: {game.headers['Result']}")
        if game.headers["Result"] == "1-0":
            if nnue_is_white: self.results["nnue_wins"] += 1
            else: self.results["stockfish_wins"] += 1
        elif game.headers["Result"] == "0-1":
            if not nnue_is_white: self.results["nnue_wins"] += 1
            else: self.results["stockfish_wins"] += 1
        else: self.results["draws"] += 1
        return game

    def run_match(self):
        all_games = []
        for i, (fen, name) in enumerate(POSITIONS):
            print(f"\n>>> Match {i*2 + 1}/{len(POSITIONS) * 2} <<<")
            game1 = self.play_game(fen, name, nnue_is_white=True)
            if game1: all_games.append(game1)
            print(f"\n>>> Match {i*2 + 2}/{len(POSITIONS) * 2} <<<")
            game2 = self.play_game(fen, name, nnue_is_white=False)
            if game2: all_games.append(game2)

        pgn_filename = f"match_nnue32_vs_sf_d{STOCKFISH_DEPTH}_{datetime.now().strftime('%Y%m%d')}.pgn"
        with open(pgn_filename, "w", encoding="utf-8") as f:
            for game in all_games:
                print(game, file=f, end="\n\n")
        print(f"\n[SUCCESS] All games saved to '{pgn_filename}'.")
        print("\n" + "="*40 + "\n           FINAL MATCH RESULTS\n" + "="*40)
        print(f"NNUE-32 Wins:    {self.results['nnue_wins']}")
        print(f"Stockfish Wins:  {self.results['stockfish_wins']}")
        print(f"Draws:           {self.results['draws']}")
        print("-"*40)
        total = sum(self.results.values())
        if total > 0:
            score = self.results['nnue_wins'] + 0.5 * self.results['draws']
            print(f"NNUE-32 Score: {score}/{total} ({(score / total) * 100:.1f}%)")
        print("="*40)

def main():
    for f in [NNUE_CHECKPOINT, BUCKET_MAPPING_FILE, STOCKFISH_PATH]:
        if not Path(f).exists():
            print(f"Error: Required file not found: '{f}'", file=sys.stderr)
            sys.exit(1)
    match = PositionalMatch(NNUE_CHECKPOINT, STOCKFISH_PATH, STOCKFISH_DEPTH)
    match.run_match()

if __name__ == "__main__":
    main()
