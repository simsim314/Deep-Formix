#!/usr/bin/env python3
"""
play_nnue32_vs_stockfish.py
Fair UCI-vs-UCI match: ./nnue32_engine vs Stockfish with symmetric depth/nodes (no time).

- No use of SimpleEngine.ucinewgame() (not present in some python-chess versions).
- Sends fresh positions each game; resets by reopening engines per game.
- Alternates colors; saves PGN with timestamp.
"""

import sys
import os
import chess
import chess.engine
import chess.pgn
from datetime import datetime
from pathlib import Path

# --------------------
# CONFIG
# --------------------
NNUE_ENGINE_PATH = "./nnue32_engine"  # your compiled C++ UCI engine
STOCKFISH_PATH   = "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"

# Choose ONE: "depth" OR "nodes"
MODE  = "depth"         # or "nodes"
DEPTH = 1               # used if MODE == "depth"
NODES = 2000            # used if MODE == "nodes"

GAMES_PER_POSITION = 2  # NNUE plays White and Black per position
SAVE_PGN = True
PGN_PREFIX = "nnue32_vs_sf"

# Keep options symmetric and modest to reduce noise
THREADS = 1
HASH_MB = 64
MULTIPV = 1

# 10 test positions (FEN, name)
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

# --------------------
# Helpers
# --------------------
def build_limit():
    if MODE == "depth":
        return chess.engine.Limit(depth=DEPTH)
    elif MODE == "nodes":
        return chess.engine.Limit(nodes=NODES)
    else:
        raise ValueError("MODE must be 'depth' or 'nodes'.")

def configure_engine(engine: chess.engine.SimpleEngine):
    """Apply symmetric, common UCI options (ignore if engine lacks an option)."""
    try:
        engine.configure({
            "Threads": THREADS,
            "Hash": HASH_MB,
            "MultiPV": MULTIPV
        })
    except Exception:
        pass  # Unknown options are fine; we keep going.

def play_single_game(fen: str, position_name: str, nnue_is_white: bool, limit: chess.engine.Limit):
    """
    Plays one game between ./nnue32_engine and Stockfish from a given FEN.
    Returns (pgn_game, result_str).
    """
    board = chess.Board(fen)
    game = chess.pgn.Game()
    tag = f"{MODE}={DEPTH if MODE=='depth' else NODES}"
    game.headers["Event"] = f"nnue32_engine vs Stockfish ({tag})"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = position_name
    game.headers["White"] = "nnue32_engine" if nnue_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if nnue_is_white else "nnue32_engine"
    game.headers["FEN"]   = fen
    game.setup(board)
    node = game

    print(f"\n--- {position_name} | {game.headers['White']} vs {game.headers['Black']} ---")

    try:
        # Open fresh engines per game to avoid state carry-over
        with chess.engine.SimpleEngine.popen_uci(NNUE_ENGINE_PATH) as nnue, \
             chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as sf:

            configure_engine(nnue)
            configure_engine(sf)

            while not board.is_game_over(claim_draw=True):
                if (board.turn == chess.WHITE) == nnue_is_white:
                    # nnue32_engine to move
                    result = nnue.play(board, limit)
                    move = result.move
                    side = "nnue32_engine"
                else:
                    # Stockfish to move
                    result = sf.play(board, limit)
                    move = result.move
                    side = "Stockfish"

                if move is None:
                    print(f"{side} returned no move, aborting this game.")
                    break

                # Uncomment to trace:
                # print(f"{side} plays {move.uci()}")

                board.push(move)
                node = node.add_variation(move)

    except FileNotFoundError as e:
        print(f"[ERROR] Engine not found: {e}", file=sys.stderr)
        game.headers["Result"] = "*"
        return game, "*"
    except Exception as e:
        print(f"[ERROR] Engine failure: {e}", file=sys.stderr)
        game.headers["Result"] = "*"
        return game, "*"

    result_str = board.result(claim_draw=True)
    game.headers["Result"] = result_str
    print(f"Game over: {result_str}")
    return game, result_str

# --------------------
# Main
# --------------------
def main():
    # Check binaries exist and are executable
    for path in [NNUE_ENGINE_PATH, STOCKFISH_PATH]:
        if not Path(path).exists():
            print(f"Error: Missing engine binary: {path}", file=sys.stderr)
            sys.exit(1)
        if not os.access(path, os.X_OK):
            print(f"Error: Engine not executable: {path}. Try: chmod +x '{path}'", file=sys.stderr)
            sys.exit(1)

    if MODE not in ("depth", "nodes"):
        print("Error: MODE must be 'depth' or 'nodes'.", file=sys.stderr)
        sys.exit(1)

    limit = build_limit()

    nnue_wins = 0
    sf_wins   = 0
    draws     = 0
    games_out = []

    total_games = len(POSITIONS) * GAMES_PER_POSITION
    idx = 0

    for fen, name in POSITIONS:
        for color_flip in range(GAMES_PER_POSITION):
            idx += 1
            nnue_is_white = (color_flip % 2 == 0)
            print(f"\n>>> Game {idx}/{total_games} ({'NNUE White' if nnue_is_white else 'NNUE Black'}) <<<")
            game, result = play_single_game(fen, name, nnue_is_white, limit)
            games_out.append(game)

            if result == "1-0":
                nnue_wins += 1 if nnue_is_white else 0
                sf_wins   += 1 if not nnue_is_white else 0
            elif result == "0-1":
                nnue_wins += 1 if not nnue_is_white else 0
                sf_wins   += 1 if nnue_is_white else 0
            elif result == "1/2-1/2":
                draws += 1
            # "*" means aborted game; don't count

    # Save PGN
    if SAVE_PGN and games_out:
        tag = f"{MODE}{DEPTH if MODE=='depth' else NODES}"
        pgn_name = f"{PGN_PREFIX}_{tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.pgn"
        with open(pgn_name, "w", encoding="utf-8") as f:
            for g in games_out:
                print(g, file=f, end="\n\n")
        print(f"\n[SAVED] PGN written to: {pgn_name}")

    # Final results
    total = nnue_wins + sf_wins + draws
    print("\n" + "=" * 42)
    print("           FINAL MATCH RESULTS")
    print("=" * 42)
    print(f"nnue32_engine Wins: {nnue_wins}")
    print(f"Stockfish Wins:     {sf_wins}")
    print(f"Draws:              {draws}")
    print("-" * 42)
    if total > 0:
        score = nnue_wins + 0.5 * draws
        pct = 100.0 * score / total
        print(f"nnue32_engine Score: {score:.1f}/{total} ({pct:.1f}%)")
    print("=" * 42)

if __name__ == "__main__":
    main()
