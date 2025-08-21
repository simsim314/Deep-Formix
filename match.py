#!/usr/bin/env python3
"""
match_cv.py — Run NNUE engine vs Stockfish (depth=4) and visualize the game in real time
from the NNUE perspective using OpenCV.

Requirements:
  pip install python-chess opencv-python

Keys:
  q or ESC  -> abort all matches immediately
"""

import sys
import os
from pathlib import Path
from contextlib import ExitStack
import tempfile
import cv2
import numpy as np
import chess
import chess.engine
from tqdm import tqdm

from positions import POSITIONS  # list of (fen, name_or_id)

# =============================================================================
# CONFIG
# =============================================================================

ENGINE_PATH        = "./engine"  # your engine, NNUE inside
STOCKFISH_PATH     = "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"
STOCKFISH_DEPTH    = 4
PIECE_DIR          = Path(__file__).resolve().parent / "Images" / "60"  # adjust if needed
SQUARE_SIZE        = 60
LIGHT_COLOR        = (240, 240, 240)  # BGR
DARK_COLOR         = (180, 200, 160)
HL_FROM_COLOR      = (120, 180, 255)  # highlight "from" square (light blue)
HL_TO_COLOR        = (80, 220, 120)   # highlight "to" square (green)
BORDER             = 20               # border for labels around the board
FONT               = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE         = 0.5
THICKNESS          = 1
LINE_TYPE          = cv2.LINE_AA

WINDOW_NAME        = "NNUE Match Viewer"
SHOW_MOVE_TEXT     = True
INIT_TIMEOUT       = 15.0  # seconds for engine init

# =============================================================================
# PIECES LOADING
# =============================================================================

def load_piece_images(piece_dir: Path, square_size: int):
    """Load PNGs with alpha and resize to square_size."""
    names = [
        "wK","wQ","wR","wB","wN","wP",
        "bK","bQ","bR","bB","bN","bP",
        "blank"
    ]
    imgs = {}
    for n in names:
        p = piece_dir / f"{n}.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing piece image: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # keep alpha
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        if (img.shape[0] != square_size) or (img.shape[1] != square_size):
            img = cv2.resize(img, (square_size, square_size), interpolation=cv2.INTER_AREA)
        imgs[n] = img
    return imgs

# Alpha blend utility
def paste_rgba(dst: np.ndarray, src_rgba: np.ndarray, x: int, y: int):
    """Alpha-blend src_rgba onto dst at (x, y). dst is BGR, src is BGRA."""
    h, w = src_rgba.shape[:2]
    roi = dst[y:y+h, x:x+w]
    b, g, r, a = cv2.split(src_rgba)
    alpha = a.astype(float) / 255.0
    inv = 1.0 - alpha
    for c, src_c in enumerate([b, g, r]):  # B, G, R
        roi[:, :, c] = (alpha * src_c + inv * roi[:, :, c]).astype(np.uint8)
    dst[y:y+h, x:x+w] = roi

# =============================================================================
# BOARD RENDERING
# =============================================================================

def board_image(board: chess.Board, imgs: dict, nnue_white: bool, last_move: chess.Move | None,
                wdl_text: str, move_text: str) -> np.ndarray:
    """
    Render the current board from NNUE perspective:
      - If nnue_white, White at bottom; else flip.
      - Highlight last move (from/to).
      - Draw WDL/score and last mover text around.
    """
    sq = SQUARE_SIZE
    board_px = 8 * sq
    # canvas with border for labels
    canvas = np.zeros((board_px + 2*BORDER + 40, board_px + 2*BORDER, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # draw squares
    top = BORDER
    left = BORDER
    img = canvas

    # mapping rank/file order from NNUE perspective
    rank_range = range(7, -1, -1) if nnue_white else range(0, 8)  # white bottom vs black bottom
    file_range = range(0, 8) if nnue_white else range(7, -1, -1)

    # last move squares for highlight (indices 0..63)
    from_sq = last_move.from_square if last_move else None
    to_sq   = last_move.to_square if last_move else None

    # prebuild square BG (light/dark) as solid color tiles
    light_tile = np.full((sq, sq, 3), LIGHT_COLOR, dtype=np.uint8)
    dark_tile  = np.full((sq, sq, 3), DARK_COLOR, dtype=np.uint8)

    # draw squares and pieces
    for rank_i, r in enumerate(rank_range):
        for file_i, f in enumerate(file_range):
            sq_index = r * 8 + f
            x = left + file_i * sq
            y = top + rank_i * sq

            color_tile = light_tile if (r + f) % 2 == 0 else dark_tile
            img[y:y+sq, x:x+sq] = color_tile

            # highlight if from/to
            if from_sq is not None and sq_index == from_sq:
                cv2.rectangle(img, (x, y), (x+sq-1, y+sq-1), HL_FROM_COLOR, 2)
            if to_sq is not None and sq_index == to_sq:
                cv2.rectangle(img, (x, y), (x+sq-1, y+sq-1), HL_TO_COLOR, 2)

            piece = board.piece_at(sq_index)
            if piece:
                key = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
                tile = imgs.get(key)
                if tile is not None:
                    paste_rgba(img, tile, x, y)

    # labels
    y_text = top + board_px + 20
    cv2.putText(img, wdl_text, (left, y_text), FONT, FONT_SCALE, (230, 230, 230), THICKNESS, LINE_TYPE)
    if SHOW_MOVE_TEXT:
        cv2.putText(img, move_text, (left, y_text + 18), FONT, FONT_SCALE, (200, 220, 255), THICKNESS, LINE_TYPE)

    # side footer
    footer = "Perspective: NNUE White" if nnue_white else "Perspective: NNUE Black"
    cv2.putText(img, footer, (left, y_text + 36), FONT, FONT_SCALE, (170, 255, 170), THICKNESS, LINE_TYPE)

    return canvas

# =============================================================================
# ENGINES
# =============================================================================

def start_engines():
    """Start fresh engines and return (my_engine, sf_engine, sf_limit)."""
    my_engine = chess.engine.SimpleEngine.popen_uci(
        [ENGINE_PATH],
        timeout=INIT_TIMEOUT  # init timeout
    )
    sf_engine = chess.engine.SimpleEngine.popen_uci([STOCKFISH_PATH], timeout=INIT_TIMEOUT)
    sf_limit = chess.engine.Limit(depth=STOCKFISH_DEPTH)
    return my_engine, sf_engine, sf_limit

def result_to_wdl(res: str, nnue_white: bool):
    if res == "1-0":
        return (1, 0, 0) if nnue_white else (0, 0, 1)
    if res == "0-1":
        return (0, 0, 1) if nnue_white else (1, 0, 0)
    return (0, 1, 0)

# =============================================================================
# GAME
# =============================================================================

def play_one_game(fen: str, nnue_white: bool, imgs: dict,
                  scoreboard_state: tuple[int,int,int]) -> tuple[int,int,int,bool]:
    """
    Play a single game from FEN.
    Visualize with OpenCV in real time from NNUE perspective.
    Returns (w,d,l, aborted)
    """
    w, d, l = scoreboard_state
    board = chess.Board(fen)
    last_move = None

    # init window once (safe to recreate each game)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    with ExitStack() as stack:
        my_engine, sf_engine, sf_limit = start_engines()
        stack.callback(my_engine.quit)
        stack.callback(sf_engine.quit)

        ply = 0
        aborted = False

        while not board.is_game_over(claim_draw=True):
            # Render current position
            total = w + d + l
            score = w + 0.5 * d
            wdl_text = f"W:{w} D:{d} L:{l}  Score:{score:.1f}/{total}" if total > 0 else "W:0 D:0 L:0"
            mover = "NNUE" if ((board.turn == chess.WHITE) == nnue_white) else f"SF(d{STOCKFISH_DEPTH})"
            move_text = f"To move: {mover}   Ply: {ply}"
            frame = board_image(board, imgs, nnue_white, last_move, wdl_text, move_text)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # ESC or q
                aborted = True
                break

            # Decide and play move
            if (board.turn == chess.WHITE) == nnue_white:
                chosen = my_engine.play(board, chess.engine.Limit())  # NNUE decides its own time/depth
            else:
                chosen = sf_engine.play(board, sf_limit)

            if chosen.move is None:
                break

            board.push(chosen.move)
            last_move = chosen.move
            ply += 1

            # Show after the move too
            total = w + d + l
            score = w + 0.5 * d
            wdl_text = f"W:{w} D:{d} L:{l}  Score:{score:.1f}/{total}" if total > 0 else "W:0 D:0 L:0"
            mover = "NNUE" if ((board.turn == chess.WHITE) == nnue_white) else f"SF(d{STOCKFISH_DEPTH})"
            move_text = f"Last: {last_move.uci()}   Next: {mover}   Ply: {ply}"
            frame = board_image(board, imgs, nnue_white, last_move, wdl_text, move_text)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                aborted = True
                break

        if aborted:
            return w, d, l, True

        # Game ended: update WDL
        res = board.result(claim_draw=True)
        gw, gd, gl = result_to_wdl(res, nnue_white)
        return w + gw, d + gd, l + gl, False

# =============================================================================
# MAIN
# =============================================================================

def main():
    if not Path(ENGINE_PATH).exists():
        print(f"Missing engine at '{ENGINE_PATH}'", file=sys.stderr)
        sys.exit(1)
    if not Path(STOCKFISH_PATH).exists():
        print(f"Missing Stockfish at '{STOCKFISH_PATH}'", file=sys.stderr)
        sys.exit(1)

    imgs = load_piece_images(PIECE_DIR, SQUARE_SIZE)

    total_games = len(POSITIONS) * 2
    w = d = l = 0

    with tqdm(total=total_games, unit="game") as pbar:
        for fen, _name in POSITIONS:
            # NNUE as White
            w, d, l, aborted = play_one_game(fen, True, imgs, (w, d, l))
            played = w + d + l
            score = w + 0.5 * d
            pbar.set_description(f"W:{w} D:{d} L:{l} Score:{score:.1f}/{played}")
            pbar.update(1)
            if aborted:
                break

            # NNUE as Black
            w, d, l, aborted = play_one_game(fen, False, imgs, (w, d, l))
            played = w + d + l
            score = w + 0.5 * d
            pbar.set_description(f"W:{w} D:{d} L:{l} Score:{score:.1f}/{played}")
            pbar.update(1)
            if aborted:
                break

    # Final frame persists until a keypress
    total = w + d + l
    score = w + 0.5 * d
    pct = (score / total) * 100 if total else 0.0
    print(f"\nFinal: W:{w} D:{d} L:{l}  {score:.1f}/{total} ({pct:.1f}%)")
    print(f"Stockfish depth: {STOCKFISH_DEPTH}")
    print("Close the OpenCV window to exit.")

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
