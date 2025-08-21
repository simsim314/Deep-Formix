#!/usr/bin/env python3
"""
depth_match.py — Stockfish depth-vs-depth mini-tournament with custom starting positions.

- Uses your STOCKFISH_PATH (edit below if needed).
- Imports POSITIONS = [(fen: str, name: str), ...] from positions.py
- Runs matches for every depth ordered pair in {2..8} without reversed duplicates
  (i.e., 2 vs 3 exists, 3 vs 2 does not), and still plays both color assignments
  per position for fairness.
- Shows tqdm progress bars per pair with W/D/L and running score.
- Computes Elo ratings for each depth from all games (iterative Elo).
"""

from __future__ import annotations
import sys
import os
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable

import chess
import chess.engine
from tqdm import tqdm

# === CONFIG ===
STOCKFISH_PATH  = "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"
DEPTHS          = list(range(2, 9))  # 2..8 inclusive
THREADS         = 1                  # safer for reproducibility
HASH_MB         = 64                 # keep it modest unless you want bigger
MOVE_OVERHEAD_MS= 50                 # engine overhead buffer (unused; can wire to time control)
PER_MOVE_LIMIT  = None               # use pure depth; leave None
MAX_PLIES       = 512                # hard cap just in case of engine loops
SEED            = 42                 # shuffle deterministically in Elo fitting
K_FACTOR        = 8.0                # Elo update step
ELO_EPOCHS      = 250                # iterations over the result set
ELO_BASE        = 400.0              # 400 elo step in logistic
SHOW_GAME_LOGS  = False              # set True to echo every game line

# Import your positions list
try:
    from positions import POSITIONS  # list of (fen: str, name: str)
except Exception as e:
    print("ERROR: Could not import POSITIONS from positions.py")
    raise

@dataclass
class GameResult:
    white_depth: int
    black_depth: int
    result_white: float  # 1.0 win, 0.5 draw, 0.0 loss

@dataclass
class PairStats:
    w: int = 0
    d: int = 0
    l: int = 0
    pts: float = 0.0  # from left depth’s perspective (White in game A, Black in game B)

def ensure_stockfish_exists(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stockfish not found at: {path}")
    if not os.access(path, os.X_OK):
        # Try making it executable, if possible
        try:
            os.chmod(path, 0o755)
        except Exception:
            pass
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Stockfish is not executable: {path}")

def uci_engine(path: str) -> chess.engine.SimpleEngine:
    """Launch a Stockfish process with safe defaults."""
    engine = chess.engine.SimpleEngine.popen_uci(path)
    # Set UCI options
    try_set = {
        "Threads": THREADS,
        "Hash": HASH_MB,
        # Keep default strength; we want pure depth play
        "UCI_LimitStrength": False,
        # "Skill Level": 20,
        # "Ponder": False,
    }
    for k, v in try_set.items():
        try:
            engine.configure({k: v})
        except Exception:
            # Not all Stockfish builds accept all options; ignore gracefully.
            pass
    return engine

def play_one_game(
    engine_white: chess.engine.SimpleEngine,
    engine_black: chess.engine.SimpleEngine,
    fen: str,
    depth_white: int,
    depth_black: int,
) -> float:
    """
    Plays a single game from FEN. Returns result from White's perspective:
    1.0 (White win), 0.5 (draw), 0.0 (White loss).
    """
    board = chess.Board(fen)
    limit_white = chess.engine.Limit(depth=depth_white)
    limit_black = chess.engine.Limit(depth=depth_black)

    for _ in range(MAX_PLIES):
        if board.is_game_over():
            break

        engine = engine_white if board.turn == chess.WHITE else engine_black
        limit  = limit_white if board.turn == chess.WHITE else limit_black

        try:
            info = engine.play(board, limit, info=chess.engine.INFO_NONE)
        except chess.engine.EngineTerminatedError:
            return 0.5
        except chess.engine.EngineError:
            return 0.5

        if info.move is None or info.move not in board.legal_moves:
            return 0.5

        board.push(info.move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0

def iterate_pairs(depths: Iterable[int]) -> List[Tuple[int, int]]:
    # >>> ONE-LINE FIX FOR ORDERED UNIQUE PAIRS (no reversed duplicates) <<<
    dlist = list(depths)
    pairs: List[Tuple[int, int]] = []
    for i in range(len(dlist)):
        for j in range(i + 1, len(dlist)):  # <— changed from range(len(dlist)) to range(i+1, ...)
            pairs.append((dlist[i], dlist[j]))
    return pairs

def run_matches() -> Tuple[List[GameResult], Dict[Tuple[int, int], PairStats]]:
    results: List[GameResult] = []
    pair_stats: Dict[Tuple[int,int], PairStats] = {}

    ensure_stockfish_exists(STOCKFISH_PATH)

    pairs = iterate_pairs(DEPTHS)
    total_positions = len(POSITIONS)

    for dW, dB in pairs:
        pair_key = (dW, dB)
        pair_stats[pair_key] = PairStats()

        engW = uci_engine(STOCKFISH_PATH)
        engB = uci_engine(STOCKFISH_PATH)
        try:
            # For this pair, each position is played twice:
            #   Game A: FEN as given (White depth=dW, Black depth=dB)
            #   Game B: Same FEN, swap colors (White depth=dB, Black depth=dW)
            n_games_in_pair = total_positions * 2

            bar = tqdm(
                total=n_games_in_pair,
                desc=f"Depth {dW} vs {dB}",
                leave=True
            )

            for fen, pos_name in POSITIONS:
                # Game A
                score_w = play_one_game(engW, engB, fen, dW, dB)
                results.append(GameResult(white_depth=dW, black_depth=dB, result_white=score_w))
                if score_w == 1.0:
                    pair_stats[pair_key].w += 1
                elif score_w == 0.5:
                    pair_stats[pair_key].d += 1
                else:
                    pair_stats[pair_key].l += 1
                pair_stats[pair_key].pts += score_w
                bar.set_postfix(W=pair_stats[pair_key].w, D=pair_stats[pair_key].d, L=pair_stats[pair_key].l, Pts=f"{pair_stats[pair_key].pts:.1f}")
                bar.update(1)
                if SHOW_GAME_LOGS:
                    print(f"[{pos_name}] {dW} (White) vs {dB} (Black): {score_w}")

                # Game B (swap colors)
                score_w_swapped = play_one_game(engB, engW, fen, dB, dW)
                results.append(GameResult(white_depth=dB, black_depth=dW, result_white=score_w_swapped))

                # Convert to points for the left depth (dW) in this pair:
                pts_for_dW = 1.0 - score_w_swapped
                if score_w_swapped == 1.0:
                    # Black (dW) lost
                    pair_stats[pair_key].l += 1
                elif score_w_swapped == 0.5:
                    pair_stats[pair_key].d += 1
                else:
                    pair_stats[pair_key].w += 1
                pair_stats[pair_key].pts += pts_for_dW
                bar.set_postfix(W=pair_stats[pair_key].w, D=pair_stats[pair_key].d, L=pair_stats[pair_key].l, Pts=f"{pair_stats[pair_key].pts:.1f}")
                bar.update(1)
                if SHOW_GAME_LOGS:
                    print(f"[{pos_name}] {dB} (White) vs {dW} (Black): {score_w_swapped} (=> pts for depth {dW}: {pts_for_dW})")

            bar.close()
        finally:
            try:
                engW.quit()
            except Exception:
                pass
            try:
                engB.quit()
            except Exception:
                pass

    return results, pair_stats

def fit_elo_from_results(results: List[GameResult], depths: Iterable[int]) -> Dict[int, float]:
    """Iterative Elo across multi-player results using all white-perspective scores."""
    depths = list(depths)
    ratings: Dict[int, float] = {d: 0.0 for d in depths}  # mean ~0 start

    matches: List[Tuple[int, int, float]] = []
    for gr in results:
        matches.append((gr.white_depth, gr.black_depth, gr.result_white))

    random.seed(SEED)
    for _ in range(ELO_EPOCHS):
        random.shuffle(matches)
        for A, B, sA in matches:
            Ra = ratings[A]
            Rb = ratings[B]
            expA = 1.0 / (1.0 + 10 ** ((Rb - Ra) / ELO_BASE))
            delta = K_FACTOR * (sA - expA)
            ratings[A] = Ra + delta
            ratings[B] = Rb - delta

    # Normalize to mean 0
    mean = sum(ratings.values()) / len(ratings)
    for d in ratings:
        ratings[d] -= mean

    return ratings

def summarize(pair_stats: Dict[Tuple[int,int], PairStats], elo: Dict[int, float],
              anchor_depth: int = 8, anchor_value: float = 2000.0) -> None:
    """
    Print pair WDL/points and Elo table. Elo values are shifted so that
    `anchor_depth` displays as `anchor_value` if the anchor depth exists.
    """
    # Compute anchored ELOs
    if anchor_depth in elo:
        offset = anchor_value - elo[anchor_depth]
    else:
        offset = 0.0  # if anchor is missing, don't shift
    anchored_elo = {d: e + offset for d, e in elo.items()}

    # Order depths by anchored Elo (desc)
    depths_sorted = sorted(anchored_elo.keys(), key=lambda d: -anchored_elo[d])
    all_depths = sorted(anchored_elo.keys())

    # Pair matrix (from left depth’s perspective)
    print("\n=== Pair WDL/Points (from left depth’s perspective) ===")
    header = "     " + "  ".join(f"d{d:>2}" for d in all_depths)
    print(header)
    for dW in all_depths:
        row = [f"d{dW:>2}"]
        for dB in all_depths:
            if dW == dB:
                row.append("  --  ")
            else:
                ps = pair_stats.get((dW, dB))
                if not ps:
                    row.append("   .   ")
                else:
                    row.append(f"{ps.w}/{ps.d}/{ps.l} ({ps.pts:.1f})")
        print(" ".join(f"{c:>10}" for c in row))

    # Elo table (anchored)
    print("\n=== Elo Ratings by Depth (anchored) ===")
    print(f"{'Depth':>5} | {'Elo':>8}")
    print("-" * 18)
    for d in depths_sorted:
        print(f"{d:>5} | {anchored_elo[d]:>8.1f}")


def main():
    ensure_stockfish_exists(STOCKFISH_PATH)
    if not POSITIONS or not isinstance(POSITIONS, list):
        raise ValueError("POSITIONS must be a non-empty list of (fen, name).")

    print(f"Found {len(POSITIONS)} starting positions.")
    print(f"Depths: {DEPTHS}")
    print("Starting matches...\n")

    results, pair_stats = run_matches()
    print("\nAll matches completed. Fitting Elo...")
    elo = fit_elo_from_results(results, DEPTHS)
    summarize(pair_stats, elo)
    print("\nDone.")

if __name__ == "__main__":
    main()
