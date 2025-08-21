#!/usr/bin/env python3
# show_positions.py
# View positions from positions.POSITIONS, advance with SPACE.

import sys
from pathlib import Path
import cv2
import numpy as np
import chess
from positions import POSITIONS  # list of (fen, name) or just fen strings

# ---------- CONFIG ----------
SQUARE_SIZE = 60
PIECE_DIR   = Path(__file__).resolve().parent / "Images" / "60"
LIGHT_COLOR = (238, 238, 210)  # BGR
DARK_COLOR  = (118, 150, 86)   # BGR
BORDER      = 20
WINDOW_NAME = "Positions Viewer"

class ChessBoard:
    def __init__(self, piece_dir: Path, square_size: int = 60):
        self.piece_dir = Path(piece_dir)
        self.sq = int(square_size)
        self._imgs = self._load_piece_images()

    def _load_piece_images(self):
        names = [
            "wK","wQ","wR","wB","wN","wP",
            "bK","bQ","bR","bB","bN","bP"
        ]
        imgs = {}
        for n in names:
            p = self.piece_dir / f"{n}.png"
            if not p.exists():
                raise FileNotFoundError(f"Missing piece image: {p}")
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image: {p}")
            if img.shape[0] != self.sq or img.shape[1] != self.sq:
                img = cv2.resize(img, (self.sq, self.sq), interpolation=cv2.INTER_AREA)
            imgs[n] = img
        return imgs

    @staticmethod
    def _coords(square: int, white_view: bool):
        r = chess.square_rank(square)
        f = chess.square_file(square)
        if white_view:
            row = 7 - r
            col = f
        else:
            row = r
            col = 7 - f
        return row, col

    @staticmethod
    def _paste_rgba(dst, src_rgba, x, y):
        h, w = src_rgba.shape[:2]
        roi = dst[y:y+h, x:x+w]
        b, g, r, a = cv2.split(src_rgba)
        alpha = a.astype(float) / 255.0
        inv = 1.0 - alpha
        for c, src_c in enumerate([b, g, r]):
            roi[:, :, c] = (alpha * src_c + inv * roi[:, :, c]).astype(np.uint8)
        dst[y:y+h, x:x+w] = roi

    def render(self, fen: str, white_view: bool, caption: str = "") -> np.ndarray:
        board = chess.Board(fen)
        sq = self.sq
        board_px = 8 * sq
        img = np.zeros((board_px + 2*BORDER + 30, board_px + 2*BORDER, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)

        light = np.full((sq, sq, 3), LIGHT_COLOR, dtype=np.uint8)
        dark  = np.full((sq, sq, 3), DARK_COLOR, dtype=np.uint8)

        top = left = BORDER

        # squares
        for s in chess.SQUARES:
            r, c = self._coords(s, white_view)
            x = left + c * sq
            y = top + r * sq
            tile = light if ((chess.square_rank(s) + chess.square_file(s)) % 2 == 0) else dark
            img[y:y+sq, x:x+sq] = tile

        # pieces
        for s in chess.SQUARES:
            piece = board.piece_at(s)
            if not piece:
                continue
            key = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
            tile = self._imgs[key]
            r, c = self._coords(s, white_view)
            x = left + c * sq
            y = top + r * sq
            self._paste_rgba(img, tile, x, y)

        # caption
        if caption:
            cv2.putText(img, caption, (left, top + 8*sq + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        return img

def _normalize_positions():
    """Return list of (fen, label). Accept (fen) or (fen, label)."""
    out = []
    for i, item in enumerate(POSITIONS):
        if isinstance(item, str):
            fen, label = item, f"#{i+1}"
        else:
            fen = item[0]
            label = item[1] if len(item) > 1 else f"#{i+1}"
        out.append((fen, label))
    return out

def main():
    viewer = ChessBoard(PIECE_DIR, SQUARE_SIZE)
    positions = _normalize_positions()
    if not positions:
        print("No positions found in positions.POSITIONS")
        sys.exit(1)

    idx = 0
    white_view = True  # press 'f' to flip

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        fen, label = positions[idx]
        caption = f"{label}  ({idx+1}/{len(positions)})  view={'White' if white_view else 'Black'}"
        frame = viewer.render(fen, white_view=white_view, caption=caption)
        cv2.imshow(WINDOW_NAME, frame)

        # wait for a key; SPACE -> next, 'f' -> flip, q/ESC -> quit
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q'), ord('Q')):  # ESC or q
            break
        elif k == 32:  # SPACE
            idx = (idx + 1) % len(positions)
        elif k in (ord('f'), ord('F')):
            white_view = not white_view
        elif k in (ord('p'), ord('P')):  # optional: previous
            idx = (idx - 1) % len(positions)
        # otherwise, ignore and re-render same position

        # handle window closed
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
