# inside your script (replace the existing startup + play_game loop)

import subprocess
import sys
import cv2
import chess
from show_board import ChessBoard
from pathlib import Path
from positions import POSITIONS  # list of (fen, name)
import time

# ==============================================================================
# SCRIPT PARAMETERS - CONFIGURE YOUR PATHS AND SETTINGS HERE
# ==============================================================================

ENGINE_PATH = "./engine"

# --- NEW PARAMETER ---
# Set the network name here.
# - To use a custom network (e.g., "nets/my_net.bin"), set this to "my_net".
# - To use the engine's compiled-in default, set this to None.
NET_NAME    = "nnuew_resnet_step_01220775"  # <--- EDIT THIS LINE

PIECE_DIR   = Path(__file__).resolve().parent / "Images" / "60"
SQUARE_SIZE = 60
WINDOW_NAME = "Engine Step-Through"
MAX_POSITIONS = 50  # play first 50 positions
READLINE_TIMEOUT = 10.0  # seconds (safety)

# ==============================================================================

def send_cmd(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def wait_for_prefix(proc, prefix, timeout=READLINE_TIMEOUT):
    """Read lines until one starts with prefix. Return that line (str) or None on timeout/EOF."""
    deadline = time.time() + timeout
    while True:
        if time.time() > deadline:
            return None
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        print("ENG:", line)
        if line.startswith(prefix):
            return line

def init_engine(proc):
    # UCI handshake — recommended
    send_cmd(proc, "uci")
    if wait_for_prefix(proc, "uciok") is None:
        raise RuntimeError("Engine did not respond with uciok")

    send_cmd(proc, "isready")
    if wait_for_prefix(proc, "readyok") is None:
        raise RuntimeError("Engine did not respond with readyok")

    send_cmd(proc, "ucinewgame")
    # some engines don't reply to ucinewgame -> proceed

def play_game(proc, fen, viewer):
    """Play a single game from FEN, stepping with SPACE until mate/draw."""
    board = chess.Board(fen)

    # tell engine the starting position and ensure it's ready
    send_cmd(proc, f"position fen {fen}")
    send_cmd(proc, "isready")
    if wait_for_prefix(proc, "readyok") is None:
        print("Warning: engine not ready after position")

    while not board.is_game_over(claim_draw=True):
        # Show board
        img = viewer.render(board.fen(), white_view=True)
        cv2.imshow(WINDOW_NAME, img)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):  # ESC/q quit all
            return True
        if key != 32:  # only SPACE triggers move
            continue

        # Ask engine for move
        send_cmd(proc, "go")

        # Wait for bestmove line
        bestline = wait_for_prefix(proc, "bestmove", timeout=READLINE_TIMEOUT)
        if bestline is None:
            print("Engine did not respond with bestmove (timeout or EOF).")
            return True

        # parse bestmove
        parts = bestline.split()
        if len(parts) >= 2 and parts[1] != "(none)":
            uci_move = parts[1]
            try:
                move = chess.Move.from_uci(uci_move)
            except Exception as e:
                print("Invalid UCI move from engine:", uci_move, "error:", e)
                return True

            if move in board.legal_moves:
                board.push(move)
                print("Applied", board.fen())
            else:
                # engine suggested illegal move — report and abort game
                print("Engine suggested illegal move:", uci_move)
                return True
        else:
            print("Engine returned bestmove (none) or malformed.")
            return True

        # Keep engine and client in sync (optional but safe)
        send_cmd(proc, f"position fen {board.fen()}")
        send_cmd(proc, "isready")
        if wait_for_prefix(proc, "readyok", timeout=2.0) is None:
            # not fatal — continue
            print("Warning: engine not ready after resync")

    # show final position briefly
    img = viewer.render(board.fen(), white_view=True)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(500)
    return False

def main():
    if not Path(ENGINE_PATH).exists():
        sys.exit(f"Missing engine: {ENGINE_PATH}")

    # Build the command to launch the engine based on the NET_NAME parameter
    engine_command = [ENGINE_PATH]
    if NET_NAME:
        engine_command.append(NET_NAME)
        print(f"Starting engine with network: {NET_NAME}")
    else:
        print("Starting engine with default compiled network.")

    # Start engine process
    proc = subprocess.Popen(
        engine_command,  # Use the command with the network name if provided
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        init_engine(proc)
    except Exception as e:
        proc.terminate()
        raise

    viewer = ChessBoard(PIECE_DIR, SQUARE_SIZE)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    quit_flag = False
    for idx, (fen, name) in enumerate(POSITIONS[:MAX_POSITIONS], start=1):
        print(f"\n=== Game {idx}/{MAX_POSITIONS}: {name} ===")
        if play_game(proc, fen, viewer):
            quit_flag = True
            break

    proc.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
