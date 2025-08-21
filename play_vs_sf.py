#!/usr/bin/env python3
import sys
import os
import subprocess
import select
import time
import cv2
import chess
import chess.engine
from show_board import ChessBoard
from pathlib import Path

# ==============================================================================
# SCRIPT PARAMETERS - CONFIGURE YOUR PATHS AND SETTINGS HERE
# ==============================================================================

ENGINE_PATH     = "./engine"  # Your NNUE UCI engine
STOCKFISH_PATH  = "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"

# --- NEW PARAMETER ---
# Set the network name here.
# - To use a custom network (e.g., "nets/my_net.bin"), set this to "my_net".
# - To use the engine's compiled-in default, set this to None.
NET_NAME        = "nnuew_resnet_step_01220775" # <--- EDIT THIS LINE

SFP_DEPTH       = 7           # Stockfish reply depth
PIECE_DIR       = Path(__file__).resolve().parent / "Images" / "60"
SQUARE_SIZE     = 60
WINDOW_NAME     = "Engine Step-Through (NNUE vs Stockfish)"
NNUE_PLAYS_WHITE = True

# ==============================================================================

# -------------------------
# Helpers for I/O printing
# -------------------------

def send_cmd(proc: subprocess.Popen, cmd: str) -> None:
    """Send a single command line to the NNUE engine, with newline and flush."""
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def read_until_bestmove(proc: subprocess.Popen) -> str | None:
    """
    Read and print NNUE engine stdout lines until a 'bestmove' appears.
    Returns the bestmove string (e.g., 'e2e4') or None if missing.
    """
    best = None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if line:
            print(line)
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] != "(none)":
                best = parts[1]
            break
    return best

def read_available_output(proc: subprocess.Popen, drain_time: float = 0.25) -> None:
    """
    Non-blocking drain of any immediate NNUE engine output.
    """
    end_time = time.time() + drain_time
    fd = proc.stdout.fileno()
    buf = ""
    while time.time() < end_time:
        rlist, _, _ = select.select([fd], [], [], max(0.0, end_time - time.time()))
        if not rlist:
            break
        chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
        if not chunk:
            break
        buf += chunk
        while True:
            if "\n" not in buf:
                break
            line, buf = buf.split("\n", 1)
            if line:
                print(line)
    if buf.strip():
        print(buf.strip())

def main():
    # Sanity checks
    if not Path(ENGINE_PATH).exists():
        sys.exit(f"Missing engine: {ENGINE_PATH}")
    if not Path(STOCKFISH_PATH).exists():
        sys.exit(f"Missing Stockfish: {STOCKFISH_PATH}")

    # Build the command to launch the engine based on the NET_NAME parameter
    engine_command = [ENGINE_PATH]
    if NET_NAME:
        engine_command.append(NET_NAME)
        print(f"Starting engine with network: {NET_NAME}")
    else:
        print("Starting engine with default compiled network.")

    # Start NNUE engine
    proc = subprocess.Popen(
        engine_command,  # Use the command with the network name if provided
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Start Stockfish (python-chess)
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Game state & UI
    board = chess.Board()
    viewer = ChessBoard(PIECE_DIR, SQUARE_SIZE)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Fresh game for NNUE engine
    send_cmd(proc, "ucinewgame")

    while not board.is_game_over(claim_draw=True):
        # Draw current position
        img = viewer.render(board.fen(), white_view=True)
        cv2.imshow(WINDOW_NAME, img)

        # Wait for SPACE (step), q/ESC to quit
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        if key != 32:
            continue

        # Decide whose turn it is
        nnue_to_move = (board.turn == chess.WHITE) == NNUE_PLAYS_WHITE

        if nnue_to_move:
            # NNUE moves
            send_cmd(proc, "go")
            best = read_until_bestmove(proc)
            if best is None:
                print("Engine returned no move; stopping.")
                break
            try:
                mv = chess.Move.from_uci(best)
            except ValueError:
                print(f"Bad UCI move from engine: {best}")
                break
            if mv in board.legal_moves:
                board.push(mv)
            else:
                print(f"Illegal move from engine: {best}")
                break
        else:
            # Stockfish replies
            result = sf.play(board, chess.engine.Limit(depth=SFP_DEPTH))
            if result.move is None:
                print("Stockfish returned no move; stopping.")
                break
            board.push(result.move)
            send_cmd(proc, f"move {result.move.uci()}")
            read_available_output(proc, drain_time=0.25)

    # Cleanup
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        sf.quit()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
