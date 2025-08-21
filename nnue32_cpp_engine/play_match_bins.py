#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import chess
import chess.engine
from tqdm import tqdm
from positions import POSITIONS  # Import positions from your positions.py

# ============================================================================
#  CONFIGURATION
# ============================================================================

NNUE_ENGINE_PATH = "./nnue32_engine"
STOCKFISH_PATH   = "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"
STRONG_NETS_DIR  = Path("strong_nets_bins")
MATCH_DEPTH      = 1

# ============================================================================
#  MATCH RUNNER
# ============================================================================

def quiet_handles():
    devnull = open(os.devnull, "w")
    return devnull, devnull

class PositionalMatch:
    def __init__(self, net_bin_path):
        self.net_bin_path = net_bin_path
        self.w = self.d = self.l = 0
        self.stockfish_limit = chess.engine.Limit(depth=MATCH_DEPTH)

        # Start engines once for this net
        devout, deverr = quiet_handles()
        try:
            with redirect_stdout(devout), redirect_stderr(deverr):
                self.nnue = chess.engine.SimpleEngine.popen_uci([NNUE_ENGINE_PATH, str(net_bin_path)])
                self.stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        finally:
            devout.close(); deverr.close()

    def close(self):
        self.nnue.quit()
        self.stockfish.quit()

    def play_game(self, fen, nnue_white):
        # --- NEW: reset engine internal state so previous-game positions do NOT count for repetition ---
        # This sends the UCI "ucinewgame" command to both engines, clearing internal repetition tables.
        try:
            # Some engine versions return nothing; ignore exceptions conservatively
            self.nnue.ucinewgame()
        except Exception:
            pass
        try:
            self.stockfish.ucinewgame()
        except Exception:
            pass
        # ------------------------------------------------------------------------------

        board = chess.Board(fen)
        while not board.is_game_over(claim_draw=True):
            if (board.turn == chess.WHITE) == nnue_white:
                move = self.nnue.play(board, self.stockfish_limit).move
            else:
                move = self.stockfish.play(board, self.stockfish_limit).move
            if move is None:
                break
            board.push(move)

        result = board.result(claim_draw=True)
        if result == "1-0":
            if nnue_white: self.w += 1
            else: self.l += 1
        elif result == "0-1":
            if not nnue_white: self.w += 1
            else: self.l += 1
        else:
            self.d += 1

    def run(self):
        games = len(POSITIONS) * 2
        with tqdm(total=games, unit="game") as pbar:
            for fen, _ in POSITIONS:
                # NNUE as White
                self.play_game(fen, True)
                total = self.w + self.d + self.l
                score = self.w + 0.5 * self.d
                pbar.set_description(f"W:{self.w} D:{self.d} L:{self.l} Score:{score:.1f}/{total}")
                pbar.update(1)

                # NNUE as Black
                self.play_game(fen, False)
                total = self.w + self.d + self.l
                score = self.w + 0.5 * self.d
                pbar.set_description(f"W:{self.w} D:{self.d} L:{self.l} Score:{score:.1f}/{total}")
                pbar.update(1)

        total = self.w + self.d + self.l
        score = self.w + 0.5 * self.d
        pct = (score / total) * 100
        return self.w, self.d, self.l, score, total, pct

# ============================================================================
#  MAIN
# ============================================================================

def main():
    if not Path(STOCKFISH_PATH).exists():
        print(f"Missing Stockfish at '{STOCKFISH_PATH}'", file=sys.stderr)
        sys.exit(1)
    if not STRONG_NETS_DIR.exists():
        print(f"Missing nets dir '{STRONG_NETS_DIR}'", file=sys.stderr)
        sys.exit(1)

    nets = sorted(p for p in STRONG_NETS_DIR.glob("*.bin") if not p.name.endswith("_mapping.bin"))
    if not nets:
        print(f"No valid .bin nets found in '{STRONG_NETS_DIR}'.", file=sys.stderr)
        sys.exit(1)

    for net in nets:
        print(f"\n=== {net.name} ===")
        match = PositionalMatch(net)
        try:
            w, d, l, score, total, pct = match.run()
        finally:
            match.close()
        print(f"{net.name}: {w}/{d}/{l}  {score:.1f}/{total} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
